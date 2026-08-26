#!/usr/bin/env python3
import os
import sys
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict

import torch
from datasets import load_dataset
import wandb

from tokenizers import Tokenizer
from transformers import (
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    set_seed,
    EarlyStoppingCallback
)

try:
    import sentencepiece as spm
except ImportError:
    spm = None

class WandbCallback(TrainerCallback):
    """Custom callback to log metrics to Weights & Biases"""
    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if state.is_world_process_zero and logs is not None:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    wandb.log({k: v}, step=state.global_step)

class HubSubfolderCallback(TrainerCallback):
    def __init__(self, repo_id, path_in_repo, language, tokenizer_type):
        self.repo_id = repo_id
        self.path_in_repo = path_in_repo
        self.language = language
        self.tokenizer_type = tokenizer_type
        self.last_epoch = None
        self.best_epoch = None
        
    def on_save(self, args, state, control, **kwargs):
        if int(os.environ.get("RANK", -1)) in [-1, 0]:
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                
                epoch = int(round(state.epoch))
                step = state.global_step
                ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
                
                if not os.path.exists(ckpt_dir):
                    return
                
                target_path = f"{self.path_in_repo}/checkpoint_{epoch}"
                api.upload_folder(
                    folder_path=ckpt_dir,
                    repo_id=self.repo_id,
                    path_in_repo=target_path,
                    repo_type="model",
                    commit_message=f"Auto-sync epoch {epoch} for {self.language} - {self.tokenizer_type}"
                )
                
                current_best_epoch = epoch
                if state.best_model_checkpoint is not None:
                    try:
                        best_step = int(state.best_model_checkpoint.split("-")[-1])
                        steps_per_epoch = step / epoch if epoch > 0 else step
                        current_best_epoch = int(round(best_step / steps_per_epoch))
                    except Exception:
                        pass
                
                if self.last_epoch is not None and self.last_epoch != current_best_epoch and self.last_epoch != epoch:
                    try:
                        api.delete_folder(folder_path=f"{self.path_in_repo}/checkpoint_{self.last_epoch}", repo_id=self.repo_id, repo_type="model")
                    except Exception:
                        pass
                
                if self.best_epoch is not None and self.best_epoch != current_best_epoch and self.best_epoch != epoch and self.best_epoch != self.last_epoch:
                    try:
                        api.delete_folder(folder_path=f"{self.path_in_repo}/checkpoint_{self.best_epoch}", repo_id=self.repo_id, repo_type="model")
                    except Exception:
                        pass
                
                self.last_epoch = epoch
                self.best_epoch = current_best_epoch
                
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Failed to push checkpoint to hub: {e}")

@dataclass
class ModelArguments:
    language: str = field(default="eng", metadata={"help": "Language code (e.g., eng)"})
    model_type: str = field(default="bpe", metadata={"help": "Tokenizer type (e.g., bpe, morphbpe, boundlessbpe)"})
    vocab_size: int = field(default=10000, metadata={"help": "Vocabulary size"})
    train_data_path: str = field(default="", metadata={"help": "Path to training text dataset"})
    eval_data_path: str = field(default="", metadata={"help": "Path to validation text dataset"})
    tokenizer_dir: str = field(default="../tokenizers-bin", metadata={"help": "Directory containing tokenizers"})
    bpe_dropout: bool = field(default=False)
    bpe_dropout_prob: float = field(default=0.1)
    superbpe_base_vocab_size: int = field(default=4000)

class DummyTokenizerForCollator:
    def __init__(self, vocab_dict, pad_id, mask_id):
        self.vocab = vocab_dict
        self.pad_token_id = pad_id
        self.mask_token_id = mask_id
        self.mask_token = "[MASK]"
        self.pad_token = "[PAD]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.unk_token = "[UNK]"
        self.return_token_type_ids = False
        if vocab_dict:
            self.cls_token_id = vocab_dict.get("[CLS]")
            self.sep_token_id = vocab_dict.get("[SEP]")
            self.unk_token_id = vocab_dict.get("[UNK]")

    def __len__(self):
        if self.vocab:
            return max(self.vocab.values()) + 1
        return 10005

    def get_vocab(self):
        return self.vocab
        
    def get_special_tokens_mask(self, val, already_has_special_tokens=False):
        special_token_ids = {self.cls_token_id, self.sep_token_id, self.pad_token_id}
        return [1 if v in special_token_ids else 0 for v in val]

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.vocab.get(tokens, self.unk_token_id)
        elif isinstance(tokens, list):
            return [self.vocab.get(token, self.unk_token_id) for token in tokens]
        return self.unk_token_id

    def pad(self, encoded_inputs, return_tensors=None, padding=True, max_length=None, pad_to_multiple_of=None, **kwargs):
        import torch
        batch = {}
        for key in encoded_inputs[0].keys():
            batch[key] = [f[key] for f in encoded_inputs]
            if return_tensors == "pt":
                batch[key] = torch.tensor(batch[key])
        return batch

    def save_pretrained(self, save_directory):
        import json
        import os
        os.makedirs(save_directory, exist_ok=True)
        if self.vocab:
            with open(os.path.join(save_directory, "vocab.json"), "w") as f:
                json.dump(self.vocab, f)

class TokenizerWrapper:
    """A unified tokenizer wrapper to map sentences across our various customs to integer IDs"""
    def __init__(self, args):
        self.args = args
        self.lang = args.language
        self.model_type = args.model_type
        self.vocab_size = args.vocab_size
        self.tok_dir = args.tokenizer_dir
        
        self.sp = None
        self.hf_tokenizer = None
        self.custom_model = None
        
        self.vocab_dict = {}
        self.id_to_token = {}
        
        self._load_model()
        self._load_vocab()
        
        # Insert generic special tokens safely past the end if not present.
        # For pathpiece use the raw (pre-decode) max ID so the base is identical
        # on every rank regardless of Rust HashMap iteration order.
        if not self.vocab_dict:
            max_id = self.vocab_size - 1
        elif hasattr(self, '_pathpiece_true_max_id') and self._pathpiece_true_max_id >= 0:
            max_id = self._pathpiece_true_max_id
        else:
            max_id = max(self.vocab_dict.values())
            
        self.special_tokens = ['[CLS]', '[SEP]', '[UNK]', '[PAD]', '[MASK]']
        
        for st in self.special_tokens:
            if st not in self.vocab_dict:
                max_id += 1
                self.vocab_dict[st] = max_id
                self.id_to_token[max_id] = st
                
        self.cls_id = self.vocab_dict['[CLS]']
        self.sep_id = self.vocab_dict['[SEP]']
        self.unk_id = self.vocab_dict['[UNK]']
        self.pad_id = self.vocab_dict['[PAD]']
        self.mask_id = self.vocab_dict['[MASK]']
        
    def _load_model(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
            
        mt = self.model_type
        l = self.lang
        v = self.vocab_size
        base = self.tok_dir
        
        if mt in ['wordpiece', 'superbpe', 'morphwp', 'sslm-wp']:
            if mt == 'superbpe':
                path = os.path.join(base, f"{l}_{mt}_{self.args.superbpe_base_vocab_size}_{v}.json")
            else:
                path = os.path.join(base, f"{l}_{mt}_{v}.json")
            self.hf_tokenizer = Tokenizer.from_file(path)
            
        elif mt == 'boundlessbpe':
            boundlessbpe_dir = os.path.join(parent_dir, "boundlessbpe")
            if boundlessbpe_dir not in sys.path: sys.path.insert(0, boundlessbpe_dir)
            from boundlessbpe import FasterRegexInference
            self.custom_model = FasterRegexInference()
            self.custom_model.load(os.path.join(base, f"{l}_{mt}_{v}.model"))
            
        elif mt == 'pickybpe':
            pickybpe_dir = os.path.join(parent_dir, "picky_bpe")
            if pickybpe_dir not in sys.path: sys.path.insert(0, pickybpe_dir)
            from picky_tokenize import BPEModel
            self.custom_model = BPEModel(os.path.join(base, f"{l}_{mt}_{v}.json"))
            
        elif mt == 'pathpiece':
            import pathpiece
            self.custom_model = pathpiece.Tokenizer(os.path.join(base, f"{l}_{mt}_{v}.vocab"))
            
        elif mt == 'myte':
            from myte_tokenizer import MyteTokenizer
            self.custom_model = MyteTokenizer()
            self.custom_model.load(
                os.path.join(base, f"{l}_{mt}_{v}.json"),
                os.path.join(base, f"{l}_{mt}_morfessor_{v}.bin")
            )
            
        elif mt == 'sage':
            sage_dir = os.path.join(parent_dir, "SaGe", "src")
            if sage_dir not in sys.path: sys.path.insert(0, sage_dir)
            from sage_tokenizer.model import SaGeTokenizer
            
            vocab_bytes = []
            with open(os.path.join(base, f"{l}_{mt}_{v}.vocab"), 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip('\n')
                    if not line: continue
                    try:
                        token_bytes = bytes.fromhex(line)
                    except ValueError:
                        try:
                            token_str = json.loads(line)
                        except json.JSONDecodeError:
                            token_str = line
                        if isinstance(token_str, int):
                            token_str = str(token_str)
                        token_bytes = token_str.encode('utf-8')
                    vocab_bytes.append(token_bytes)
                    
            existing = set(vocab_bytes)
            for i in range(256):
                b = bytes([i])
                if b not in existing:
                    vocab_bytes.append(b)
            self.custom_model = SaGeTokenizer(vocab_bytes)
            
        else: # bpe, unigram, morphbpe, morphulm
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(os.path.join(base, f"{l}_{mt}_{v}.model"))

    def _load_vocab(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        if parent_dir not in sys.path: sys.path.insert(0, parent_dir)
            
        mt = self.model_type
        
        if self.hf_tokenizer is not None:
            self.vocab_dict = self.hf_tokenizer.get_vocab()
            
        elif mt == 'boundlessbpe':
            self.vocab_dict = {}
            for i in range(self.vocab_size):
                self.vocab_dict[f"__dummy_{i}__"] = i
                
        elif mt == 'pickybpe':
            self.vocab_dict = self.custom_model.token2id if hasattr(self.custom_model, 'token2id') else {}
            if not self.vocab_dict and hasattr(self.custom_model, 'id2token'):
                obj = self.custom_model.id2token
                if isinstance(obj, dict):
                    self.vocab_dict = {getattr(v, 'str', str(v)): int(k) for k, v in obj.items()}
                else:
                    self.vocab_dict = {getattr(t, 'str', str(t)): i for i, t in enumerate(obj)}
                    
        elif mt == 'pathpiece':
            # get_vocab() returns {bytes: int}; decode keys to str for consistency.
            # IMPORTANT: compute max ID from raw values BEFORE decoding, because
            # many byte sequences (e.g. \x80-\xff) all decode to the same
            # replacement char \ufffd, causing dict collisions whose resolution
            # depends on Rust HashMap's random per-process iteration order.
            # Using the post-decode max would give different values on each rank.
            raw_vocab = self.custom_model.get_vocab()
            self._pathpiece_true_max_id = max(raw_vocab.values()) if raw_vocab else -1
            self.vocab_dict = {
                k.decode('utf-8', errors='replace'): v
                for k, v in raw_vocab.items()
            }
                
        elif mt == 'myte':
            if hasattr(self.custom_model, 'token2id'):
                self.vocab_dict = self.custom_model.token2id
            
        elif mt == 'sage':
            # byte_vocab is the canonical {bytes: int} map; decode keys to str for
            # consistency with the rest of the pipeline.  Collisions from decoding
            # (e.g. multi-byte sequences that share a UTF-8 replacement char) are
            # harmless here because we only need max(values()) to be accurate.
            if hasattr(self.custom_model, 'byte_vocab'):
                self.vocab_dict = {
                    k.decode('utf-8', errors='replace'): v
                    for k, v in self.custom_model.byte_vocab.items()
                }
            
        elif self.sp is not None:
            for i in range(self.sp.get_piece_size()):
                self.vocab_dict[self.sp.id_to_piece(i)] = i
                
        self.id_to_token = {i: t for t, i in self.vocab_dict.items()}

    def encode(self, text):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        if parent_dir not in sys.path: sys.path.insert(0, parent_dir)
        
        from run_segment import get_segments_pickybpe, get_segments_myte
        
        mt = self.model_type
        
        if mt == 'boundlessbpe':
            return self.custom_model.encode_ordinary(text, blowup=True)
            
        elif mt in ['wordpiece', 'superbpe', 'morphwp', 'sslm-wp']:
            return self.hf_tokenizer.encode(text).ids
            
        elif mt == 'pickybpe':
            words = get_segments_pickybpe(self.custom_model, text)
            subwords = [sw for word in words for sw in word]
            return [self.vocab_dict.get(sw, self.unk_id) for sw in subwords]
            
        elif mt == 'myte':
            words = get_segments_myte(self.custom_model, text)
            subwords = [sw for word in words for sw in word]
            return [self.vocab_dict.get(sw, self.unk_id) for sw in subwords]
            
        elif mt == 'pathpiece':
            return self.custom_model(text)['input_ids']
            
        elif mt == 'sage':
            # tokenize(tokens_only=True) already returns List[int] token IDs
            # directly from byte_vocab — do NOT do a secondary vocab_dict lookup.
            return self.custom_model.tokenize(text, tokens_only=True)
                
        elif self.sp is not None:
            if self.args.bpe_dropout and mt in ['bpe', 'morphbpe']:
                return self.sp.encode_as_ids(text, enable_sampling=True, alpha=self.args.bpe_dropout_prob, nbest_size=-1)
            return self.sp.encode_as_ids(text)
            
        return []

def main():
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()
    
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    
    logger.info(f'Training args: {training_args}')
    logger.info(f'Model args: {args}')
    
    tokenizer_type = args.model_type
    if tokenizer_type == "bpe" and getattr(args, "bpe_dropout", False):
        tokenizer_type = "bpe-dropout"
        
    run_name = f"{args.language}_{tokenizer_type}_{args.vocab_size}"
    
    if int(os.environ.get("RANK", -1)) in [-1, 0]:
        wandb.init(
            project="bert-pretraining-p3-25M",
            name=run_name,
            entity="vemulasakethreddy_10",
            config={
                "language": args.language,
                "tokenizer_type": tokenizer_type,
                "vocab_size": args.vocab_size,
                "learning_rate": training_args.learning_rate,
                "batch_size": training_args.per_device_train_batch_size * training_args.n_gpu,
                "num_train_epochs": training_args.num_train_epochs,
            }
        )
    
    set_seed(training_args.seed)
    
    # Initialize wrapper to dynamically map text to ints
    tok_wrapper = TokenizerWrapper(args)
    
    logger.info("Tokenizing dataset...")
    dataset_train = load_dataset("text", data_files=args.train_data_path, split="train")
    if int(os.environ.get("RANK", -1)) in [-1, 0]:
        dataset_train = dataset_train.shuffle(seed=42)
        
    dataset_eval = load_dataset("text", data_files=args.eval_data_path, split="train")

    def map_tokenize(samples):
        batch_input_ids = []
        batch_attention_mask = []
        batch_special_tokens = []
        
        for text in samples['text']:
            ids = tok_wrapper.encode(text)
            
            max_len = 128
            if len(ids) > max_len - 2:
                ids = ids[:max_len - 2]
                
            input_ids = [tok_wrapper.cls_id] + ids + [tok_wrapper.sep_id]
            attention_mask = [1] * len(input_ids)
            special_tokens_mask = [1] + [0] * len(ids) + [1]
            
            if len(input_ids) < max_len:
                pad_len = max_len - len(input_ids)
                input_ids += [tok_wrapper.pad_id] * pad_len
                attention_mask += [0] * pad_len
                special_tokens_mask += [1] * pad_len
                
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_special_tokens.append(special_tokens_mask)
            
        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "special_tokens_mask": batch_special_tokens
        }
        
    try:
        import dill
        dill.dumps(tok_wrapper)
        safe_num_proc = 4
    except Exception as e:
        logger.warning("Tokenizer is not picklable. Falling back to single-process tokenization (num_proc=1).")
        safe_num_proc = 1

    if getattr(args, "bpe_dropout", False) and args.model_type in ["bpe", "morphbpe"]:
        # BPE Dropout: disable caching so each epoch gets fresh stochastic tokenization.
        # We still use .map() (not set_transform) to ensure the 'text' column is removed
        # and the dataset columns match the model's forward signature.
        logger.info("BPE Dropout enabled. Tokenizing without cache (load_from_cache_file=False).")
        dataset_train = dataset_train.map(
            map_tokenize, batched=True, num_proc=safe_num_proc,
            remove_columns=["text"], load_from_cache_file=False
        )
        dataset_eval = dataset_eval.map(
            map_tokenize, batched=True, num_proc=safe_num_proc,
            remove_columns=["text"], load_from_cache_file=False
        )
    else:
        dataset_train = dataset_train.map(map_tokenize, batched=True, num_proc=safe_num_proc, remove_columns=["text"])
        dataset_eval = dataset_eval.map(map_tokenize, batched=True, num_proc=safe_num_proc, remove_columns=["text"])
    
    collator_tokenizer = DummyTokenizerForCollator(tok_wrapper.vocab_dict, tok_wrapper.pad_id, tok_wrapper.mask_id)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=collator_tokenizer,
        mlm=True,
        mlm_probability=0.15
    )
    
    vocab_size_to_use = max(len(collator_tokenizer), args.vocab_size + 1)
    # PathPiece: use the deterministic raw max ID (captured before bytes→str
    # decoding) to compute vocab_size_to_use.  DO NOT use max(vocab_dict.values())
    # here — after decoding, collision resolution is random across ranks (Rust
    # HashMap random seeds), causing different vocab_size_to_use per rank and a
    # DDP parameter shape mismatch crash.
    if args.model_type == 'pathpiece':
        true_max = getattr(tok_wrapper, '_pathpiece_true_max_id', -1)
        if true_max >= 0:
            # +6: covers the true_max token plus 5 appended special tokens
            vocab_size_to_use = max(vocab_size_to_use, true_max + 6)
    logger.info(f"Setting safe BertConfig vocab_size: {vocab_size_to_use}")
    
    # # Small (~4.8M)
    # config = BertConfig(
    #     attention_probs_dropout_prob=0.1,
    #     hidden_act="gelu",
    #     hidden_dropout_prob=0.1,
    #     hidden_size=128,
    #     initializer_range=0.01,
    #     intermediate_size=512,
    #     layer_norm_eps=1e-12,
    #     max_position_embeddings=128,
    #     num_attention_heads=4,
    #     num_hidden_layers=3,
    #     vocab_size=vocab_size_to_use
    # )
    # # Medium (Original: ~12.4M)
    # config = BertConfig(
    #     attention_probs_dropout_prob=0.1,
    #     hidden_act="gelu",
    #     hidden_dropout_prob=0.1,
    #     hidden_size=384,
    #     initializer_range=0.02,
    #     intermediate_size=1024,
    #     layer_norm_eps=1e-12,
    #     max_position_embeddings=128,
    #     num_attention_heads=6,
    #     num_hidden_layers=6,
    #     vocab_size=vocab_size_to_use
    # )
    # Large (~30.7M)
    config = BertConfig(
        attention_probs_dropout_prob=0.1,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        hidden_size=512,
        initializer_range=0.02,
        intermediate_size=2048,
        layer_norm_eps=1e-12,
        max_position_embeddings=128,
        num_attention_heads=8,
        num_hidden_layers=8,
        vocab_size=vocab_size_to_use
    )
    
    model = BertForMaskedLM(config)
    
    if int(os.environ.get("RANK", -1)) in [-1, 0]:
        total_params = sum(p.numel() for p in model.parameters())
        wandb.log({"total_parameters": total_params})
        
    wandb_callback = WandbCallback()
    
    hub_callback = HubSubfolderCallback(
        repo_id="SakethVemula/BERT-models-25M",
        path_in_repo=f"{args.language}/{tokenizer_type}_{args.vocab_size}",
        language=args.language,
        tokenizer_type=tokenizer_type
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        callbacks=[wandb_callback, hub_callback, EarlyStoppingCallback(early_stopping_patience=7)]
    )
    
    trainer.train()
    trainer.save_model(training_args.output_dir)
    
    if int(os.environ.get("RANK", -1)) in [-1, 0]:
        wandb.finish()
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_folder(
                folder_path=training_args.output_dir,
                repo_id="SakethVemula/BERT-models-25M",
                path_in_repo=f"{args.language}/{tokenizer_type}_{args.vocab_size}/final",
                ignore_patterns=["checkpoint-*"],
                repo_type="model",
                commit_message=f"Final best model for {args.language} - {tokenizer_type}"
            )
        except Exception:
            pass
        
if __name__ == '__main__':
    main()


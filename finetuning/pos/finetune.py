import os
import sys
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, List

import numpy as np
import torch
from datasets import load_dataset
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    set_seed,
    default_data_collator
)
from tokenizers import Tokenizer

try:
    import sentencepiece as spm
except ImportError:
    spm = None

logger = logging.getLogger(__name__)

class WandbCallback(TrainerCallback):
    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Dict = None, **kwargs):
        if state.is_world_process_zero and logs is not None:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    wandb.log({k: v}, step=state.global_step)

@dataclass
class ModelArguments:
    language: str = field(default="hi", metadata={"help": "Language code"})
    model_type: str = field(default="bpe", metadata={"help": "Tokenizer type"})
    vocab_size: int = field(default=50277, metadata={"help": "Vocabulary size"})
    bpe_dropout: bool = field(default=False)
    bpe_dropout_prob: float = field(default=0.1)
    superbpe_base_vocab_size: int = field(default=4000)
    tokenizer_dir: str = field(default="../../tokenizers-bin", metadata={"help": "Directory containing tokenizers"})
    max_length: int = field(default=128, metadata={"help": "Maximum sequence length"})

@dataclass
class DataArguments:
    dataset_name: str = field(default="universal_dependencies", metadata={"help": "Dataset name"})
    dataset_config: str = field(default="hi_hdtb", metadata={"help": "Dataset config"})

class TokenizerWrapper:
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
        script_dir = os.path.dirname(os.path.abspath(__file__))
        finetuning_dir = os.path.dirname(script_dir)
        self.parent_dir = os.path.dirname(finetuning_dir)
        if self.parent_dir not in sys.path:
            sys.path.insert(0, self.parent_dir)
        self._load_model()
        self._load_vocab()
        
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
        mt = self.model_type
        l = self.lang
        v = self.vocab_size
        base = self.tok_dir
        parent_dir = self.parent_dir
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
        else:
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(os.path.join(base, f"{l}_{mt}_{v}.model"))

    def _load_vocab(self):
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
            # Capture the true max ID from raw values BEFORE decoding to avoid
            # collision-resolution non-determinism across processes.
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
            # byte_vocab is the canonical {bytes: int} map; decode keys to str.
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
        pretraining_dir = os.path.join(self.parent_dir, "pretraining")
        if pretraining_dir not in sys.path:
            sys.path.insert(0, pretraining_dir)
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
            # tokenize(tokens_only=True) returns List[int] directly from byte_vocab
            return self.custom_model.tokenize(text, tokens_only=True)
        elif self.sp is not None:
            if self.args.bpe_dropout and mt in ['bpe', 'morphbpe']:
                # turn off sampling for eval by default, but this wrapper encodes for training.
                return self.sp.encode_as_ids(text, enable_sampling=True, alpha=self.args.bpe_dropout_prob, nbest_size=-1)
            return self.sp.encode_as_ids(text)
        return []

def compute_pos_metrics(eval_preds):
    predictions, labels = eval_preds
    predictions = np.argmax(predictions, axis=2)
    true_predictions = []
    true_labels = []
    for prediction, label in zip(predictions, labels):
        true_pred = [p for (p, l) in zip(prediction, label) if l != -100]
        true_lab = [l for l in label if l != -100]
        true_predictions.extend(true_pred)
        true_labels.extend(true_lab)

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, true_predictions, average='weighted')
    accuracy = accuracy_score(true_labels, true_predictions)
    return {'accuracy': accuracy, 'f1': f1, 'precision': precision, 'recall': recall}

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    
    set_seed(training_args.seed)
    
    if training_args.local_rank == 0 and "wandb" in training_args.report_to:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "p3-pos-finetuning-8M"),
            name=f"pos_{data_args.dataset_config}_{model_args.language}_{'bpe-dropout' if model_args.bpe_dropout else model_args.model_type}_v{model_args.vocab_size}",
            entity="vemulasakethreddy_10"
        )
    
    dataset = load_dataset(data_args.dataset_name, data_args.dataset_config, trust_remote_code=True)
    unique_tags = set()
    for example in dataset["train"]["upos"]:
        unique_tags.update(example)
    label_list = sorted(list(unique_tags))
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}
    num_labels = len(label_list)
    logger.info(f"Found {num_labels} POS tags")

    tok_wrapper = TokenizerWrapper(model_args)
    
    def map_tokenize(samples):
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        for i, words in enumerate(samples["tokens"]):
            input_ids = [tok_wrapper.cls_id]
            labels = [-100]
            word_labels = samples["upos"][i]
            for word, label in zip(words, word_labels):
                word_ids = tok_wrapper.encode(word)
                if not word_ids:
                    continue
                input_ids.extend(word_ids)
                labels.append(label2id[label])
                labels.extend([-100] * (len(word_ids) - 1))
            input_ids.append(tok_wrapper.sep_id)
            labels.append(-100)
            
            max_len = model_args.max_length
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len-1] + [tok_wrapper.sep_id]
                labels = labels[:max_len-1] + [-100]
                
            attention_mask = [1] * len(input_ids)
            if len(input_ids) < max_len:
                pad_len = max_len - len(input_ids)
                input_ids += [tok_wrapper.pad_id] * pad_len
                attention_mask += [0] * pad_len
                labels += [-100] * pad_len
                
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
            
        return {"input_ids": batch_input_ids, "attention_mask": batch_attention_mask, "labels": batch_labels}

    try:
        import dill
        dill.dumps(tok_wrapper)
        safe_num_proc = 4
    except Exception as e:
        safe_num_proc = 1
        
    tokenized_dataset = dataset.map(
        map_tokenize,
        batched=True,
        num_proc=safe_num_proc,
        remove_columns=dataset["train"].column_names
    )
    
    subfolder = f"{model_args.language}/{model_args.model_type}_{model_args.vocab_size}/final"
    logger.info(f"Loading model from SakethVemula/BERT-models-8M subfolder={subfolder}")
    
    model = AutoModelForTokenClassification.from_pretrained(
        "SakethVemula/BERT-models-8M",
        subfolder=subfolder,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    # Guard: resize embedding table if the tokenizer can produce IDs beyond the
    # pretrained model's vocab_size (e.g. SaGe byte-vocab IDs were not fully
    # accounted for at pretraining time, causing CUDA OOB asserts).
    required_vocab_size = max(tok_wrapper.vocab_dict.values()) + 1
    if required_vocab_size > model.config.vocab_size:
        logger.info(
            f"Resizing embedding table: {model.config.vocab_size} -> {required_vocab_size} "
            f"to cover all tokenizer IDs"
        )
        model.resize_token_embeddings(required_vocab_size)

    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    if "wandb" in training_args.report_to:
        callbacks.append(WandbCallback())
        
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"] if "validation" in tokenized_dataset else tokenized_dataset["test"],
        compute_metrics=compute_pos_metrics,
        data_collator=default_data_collator,
        callbacks=callbacks
    )
    
    train_result = trainer.train()
    trainer.save_model(training_args.output_dir)
    
    with open(f"{training_args.output_dir}/pos_labels.json", 'w') as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

if __name__ == "__main__":
    main()

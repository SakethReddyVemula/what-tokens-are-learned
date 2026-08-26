import os
import sys
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from datasets import load_dataset
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    AutoModel,
    AutoConfig,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    set_seed
)
from tokenizers import Tokenizer

try:
    import sentencepiece as spm
except ImportError:
    spm = None

logger = logging.getLogger(__name__)

LANGUAGE_CONFIGS = {
    "eng": {"treebank": "en_ewt", "name": "English-EWT"},
    "hin": {"treebank": "hi_hdtb", "name": "Hindi-HDTB"},
    "tel": {"treebank": "te_mtg", "name": "Telugu-MTG"},
    "mal": {"treebank": "ml_ufal", "name": "Malayalam-UFAL"},
    "tam": {"treebank": "ta_ttb", "name": "Tamil-TTB"}
}

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
    arc_hidden_dim: int = field(default=500, metadata={"help": "Hidden dimension for arc prediction"})
    dropout: float = field(default=0.3, metadata={"help": "Dropout rate"})

@dataclass
class DataArguments:
    dataset_name: str = field(default="universal_dependencies", metadata={"help": "Dataset name"})


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
            return self.custom_model.tokenize(text, tokens_only=True)
        elif self.sp is not None:
            if self.args.bpe_dropout and mt in ['bpe', 'morphbpe']:
                return self.sp.encode_as_ids(text, enable_sampling=True, alpha=self.args.bpe_dropout_prob, nbest_size=-1)
            return self.sp.encode_as_ids(text)
        return []

class BiaffineDependencyParser(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = AutoModel.from_config(config)
        hidden_size = config.hidden_size
        self.arc_h = nn.Linear(hidden_size, config.arc_hidden_dim)
        self.arc_d = nn.Linear(hidden_size, config.arc_hidden_dim)
        self.arc_biaffine = nn.Parameter(torch.zeros(config.arc_hidden_dim, config.arc_hidden_dim))
        self.label_h = nn.Linear(hidden_size, config.arc_hidden_dim)
        self.label_d = nn.Linear(hidden_size, config.arc_hidden_dim)
        self.label_biaffine = nn.Parameter(torch.zeros(config.num_labels, config.arc_hidden_dim, config.arc_hidden_dim))
        self.dropout = nn.Dropout(config.dropout)
        self.init_weights()
    
    def init_weights(self):
        nn.init.xavier_uniform_(self.arc_biaffine)
        nn.init.xavier_uniform_(self.label_biaffine)
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        head_ids=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        word_starts=None,
    ):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        
        sequence_output = outputs[0]
        batch_size, seq_len, _ = sequence_output.shape
        
        if word_starts is None:
            word_starts = attention_mask.bool()
        else:
            word_starts = word_starts.bool()
        
        arc_h = self.dropout(self.arc_h(sequence_output))
        arc_d = self.dropout(self.arc_d(sequence_output))
        arc_scores = torch.einsum('bih,hj,bjk->bik', arc_d, self.arc_biaffine, arc_h.transpose(1, 2))
        
        label_h = self.dropout(self.label_h(sequence_output))
        label_d = self.dropout(self.label_d(sequence_output))
        
        batch_size, seq_len, hidden_dim = label_h.size()
        num_labels = self.config.num_labels
        
        label_scores = []
        for i in range(num_labels):
            biaffine_for_label = self.label_biaffine[i]
            h_transformed = torch.matmul(label_h, biaffine_for_label)
            scores_for_label = torch.bmm(h_transformed, label_d.transpose(1, 2))
            label_scores.append(scores_for_label)
        
        label_scores = torch.stack(label_scores, dim=-1)
        
        loss = None
        if head_ids is not None and labels is not None:
            arc_loss_mask = word_starts & (head_ids != -100)
            head_ids_masked = head_ids.clone()
            head_ids_masked[~arc_loss_mask] = 0
            
            arc_loss = F.cross_entropy(
                arc_scores.view(-1, seq_len)[arc_loss_mask.view(-1)],
                head_ids_masked.view(-1)[arc_loss_mask.view(-1)],
                reduction='mean'
            )
            
            label_loss_mask = word_starts & (labels != -100)
            labels_masked = labels.clone()
            labels_masked[~label_loss_mask] = 0
            
            gold_head_label_scores = torch.zeros(batch_size, seq_len, self.config.num_labels).to(label_scores.device)
            
            for b in range(batch_size):
                for i in range(seq_len):
                    if label_loss_mask[b, i]:
                        head_idx = head_ids_masked[b, i]
                        gold_head_label_scores[b, i] = label_scores[b, i, head_idx]
            
            label_loss = F.cross_entropy(
                gold_head_label_scores.view(-1, self.config.num_labels)[label_loss_mask.view(-1)],
                labels_masked.view(-1)[label_loss_mask.view(-1)],
                reduction='mean'
            )
            
            loss = arc_loss + label_loss
        
        return {
            "loss": loss,
            "arc_logits": arc_scores,
            "label_logits": label_scores,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
        }

@dataclass
class DataCollatorForDependencyParsing:
    tokenizer: TokenizerWrapper
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    
    def __call__(self, features):
        input_ids = [feature["input_ids"] for feature in features]
        attention_mask = [feature["attention_mask"] for feature in features]
        head_ids = [feature["head_ids"] for feature in features]
        labels = [feature["labels"] for feature in features]
        word_starts = [feature["word_starts"] for feature in features]
        
        batch = {}
        batch["input_ids"] = torch.tensor(input_ids)
        batch["attention_mask"] = torch.tensor(attention_mask)
        batch["head_ids"] = torch.tensor(head_ids)
        batch["labels"] = torch.tensor(labels)
        batch["word_starts"] = torch.tensor(word_starts)
        
        return batch

def compute_parsing_metrics(eval_pred, id2label):
    arc_preds, label_preds = eval_pred.predictions
    head_ids, labels, word_starts = eval_pred.label_ids
    
    word_starts = word_starts.astype(bool)
    
    correct_heads = 0
    correct_labels = 0
    total_tokens = 0
    
    true_labels = []
    pred_labels_list = []
    
    for i in range(head_ids.shape[0]):
        for j in range(head_ids.shape[1]):
            if word_starts[i, j] and head_ids[i, j] != -100:
                total_tokens += 1
                if arc_preds[i, j] == head_ids[i, j]:
                    correct_heads += 1
                    if label_preds[i, j] == labels[i, j]:
                        correct_labels += 1
                
                if labels[i, j] != -100:
                    true_labels.append(labels[i, j])
                    pred_labels_list.append(label_preds[i, j])
    
    uas = correct_heads / total_tokens if total_tokens > 0 else 0
    las = correct_labels / total_tokens if total_tokens > 0 else 0
    
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels_list, average='weighted', zero_division=0)
    
    return {'uas': uas, 'las': las, 'precision': precision, 'recall': recall, 'f1': f1}

class DependencyParsingTrainer(Trainer):
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        has_labels = all(inputs.get(k) is not None for k in ["head_ids", "labels"])
        batch_size = inputs["input_ids"].size(0)
        seq_len = inputs["input_ids"].size(1)
        
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs["loss"] if "loss" in outputs else None
            arc_logits = outputs["arc_logits"].detach()
            label_logits = outputs["label_logits"].detach()
            
            self._stored_seq_len = getattr(self, "_stored_seq_len", None)
            if self._stored_seq_len is None:
                self._stored_seq_len = seq_len
            
            arc_preds = torch.argmax(arc_logits, dim=-1)
            b_indices = torch.arange(batch_size, device=label_logits.device).unsqueeze(1).expand(batch_size, seq_len)
            seq_indices = torch.arange(seq_len, device=label_logits.device).unsqueeze(0).expand(batch_size, seq_len)
            selected_label_logits = label_logits[b_indices, seq_indices, arc_preds]
            label_preds = torch.argmax(selected_label_logits, dim=-1)
            
            if seq_len != self._stored_seq_len:
                if seq_len < self._stored_seq_len:
                    pad_size = self._stored_seq_len - seq_len
                    arc_preds = torch.nn.functional.pad(arc_preds, (0, pad_size), "constant", -100)
                    label_preds = torch.nn.functional.pad(label_preds, (0, pad_size), "constant", -100)
                else:
                    arc_preds = arc_preds[:, :self._stored_seq_len]
                    label_preds = label_preds[:, :self._stored_seq_len]
            
            logits = (arc_preds, label_preds)
        
        if prediction_loss_only:
            return (loss, None, None)
        
        if has_labels:
            head_ids = inputs["head_ids"].detach()
            labels = inputs["labels"].detach()
            word_starts = inputs["word_starts"].detach()
            
            if seq_len != self._stored_seq_len:
                if seq_len < self._stored_seq_len:
                    pad_size = self._stored_seq_len - seq_len
                    head_ids = torch.nn.functional.pad(head_ids, (0, pad_size), "constant", -100)
                    labels = torch.nn.functional.pad(labels, (0, pad_size), "constant", -100)
                    word_starts = torch.nn.functional.pad(word_starts, (0, pad_size), "constant", 0)
                else:
                    head_ids = head_ids[:, :self._stored_seq_len]
                    labels = labels[:, :self._stored_seq_len]
                    word_starts = word_starts[:, :self._stored_seq_len]
            
            processed_labels = (head_ids, labels, word_starts)
        else:
            processed_labels = None
        
        return (loss, logits, processed_labels)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    if model_args.language not in LANGUAGE_CONFIGS:
        raise ValueError(f"Language {model_args.language} not supported.")
    lang_config = LANGUAGE_CONFIGS[model_args.language]
    dataset_config = lang_config["treebank"]
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    
    set_seed(training_args.seed)
    
    if training_args.local_rank == 0 and "wandb" in training_args.report_to:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "p3-dependency_parsing-finetuning-8M"),
            name=f"deps_{dataset_config}_{model_args.language}_{'bpe-dropout' if model_args.bpe_dropout else model_args.model_type}_v{model_args.vocab_size}",
            entity="vemulasakethreddy_10"
        )
    
    tok_wrapper = TokenizerWrapper(model_args)
    dataset = load_dataset(data_args.dataset_name, dataset_config, trust_remote_code=True)
    
    unique_deps = set()
    for example in dataset["train"]["deprel"]:
        unique_deps.update(example)
    dep_label_list = sorted(list(unique_deps))
    dep_label2id = {label: i for i, label in enumerate(dep_label_list)}
    id2dep_label = {i: label for label, i in dep_label2id.items()}
    num_labels = len(dep_label_list)

    def map_tokenize(samples):
        all_input_ids = []
        all_attention_mask = []
        all_head_ids = []
        all_labels = []
        all_word_starts = []
        
        for i, (words, heads, deps) in enumerate(zip(samples["tokens"], samples["head"], samples["deprel"])):
            clean_heads = []
            for h in heads:
                if h == 'None' or h is None:
                    clean_heads.append(-100)
                else:
                    clean_heads.append(int(h))
            heads = clean_heads
            
            input_ids = [tok_wrapper.cls_id]
            word_starts = [0]
            
            word_idx_to_token_idx = {}
            for j, word in enumerate(words):
                subwords = tok_wrapper.encode(word)
                if not subwords:
                    word_idx_to_token_idx[j] = None
                    continue
                word_idx_to_token_idx[j] = len(input_ids)
                input_ids.extend(subwords)
                word_starts.append(1)
                word_starts.extend([0]*(len(subwords)-1))
                
            input_ids.append(tok_wrapper.sep_id)
            word_starts.append(0)
            
            head_ids = [-100] * len(input_ids)
            label_ids = [-100] * len(input_ids)
            
            for j, word in enumerate(words):
                token_idx = word_idx_to_token_idx.get(j)
                if token_idx is None: continue
                
                orig_head_idx = heads[j]
                if orig_head_idx == -100:
                    head_ids[token_idx] = -100
                    label_ids[token_idx] = -100
                    continue
                    
                if orig_head_idx == 0:
                    head_token_idx = 0
                else:
                    h_idx = orig_head_idx - 1
                    head_token_idx = word_idx_to_token_idx.get(h_idx, 0)
                    if head_token_idx is None:
                        head_token_idx = 0
                    
                head_ids[token_idx] = head_token_idx
                label_ids[token_idx] = dep_label2id.get(deps[j], 0)
                
            max_len = model_args.max_length
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len-1] + [tok_wrapper.sep_id]
                word_starts = word_starts[:max_len-1] + [0]
                head_ids = head_ids[:max_len-1] + [-100]
                label_ids = label_ids[:max_len-1] + [-100]
                
            current_len = len(input_ids)
            for k in range(current_len):
                if head_ids[k] != -100 and head_ids[k] >= current_len:
                    head_ids[k] = -100
                    label_ids[k] = -100
                
            attention_mask = [1] * len(input_ids)
            
            if len(input_ids) < max_len:
                pad_len = max_len - len(input_ids)
                input_ids += [tok_wrapper.pad_id] * pad_len
                attention_mask += [0] * pad_len
                word_starts += [0] * pad_len
                head_ids += [-100] * pad_len
                label_ids += [-100] * pad_len
                
            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_word_starts.append(word_starts)
            all_head_ids.append(head_ids)
            all_labels.append(label_ids)
            
        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "head_ids": all_head_ids,
            "labels": all_labels,
            "word_starts": all_word_starts
        }

    try:
        import dill
        dill.dumps(tok_wrapper)
        safe_num_proc = 4
    except Exception as e:
        safe_num_proc = 1
        
    tokenized_datasets = dataset.map(
        map_tokenize,
        batched=True,
        num_proc=safe_num_proc,
        remove_columns=dataset["train"].column_names
    )
    
    subfolder = f"{model_args.language}/{model_args.model_type}_{model_args.vocab_size}/final"
    logger.info(f"Loading model from SakethVemula/BERT-models-8M subfolder={subfolder}")
    
    config = AutoConfig.from_pretrained("SakethVemula/BERT-models-8M", subfolder=subfolder)
    config.num_labels = num_labels
    config.arc_hidden_dim = model_args.arc_hidden_dim
    config.dropout = model_args.dropout
    config.id2label = id2dep_label
    config.label2id = dep_label2id
    
    base_model = AutoModel.from_pretrained("SakethVemula/BERT-models-8M", subfolder=subfolder)

    # Guard: resize embedding table if the tokenizer can produce IDs beyond the
    # pretrained model's vocab_size (e.g. SaGe byte-vocab IDs were not fully
    # accounted for at pretraining time, causing CUDA OOB asserts).
    required_vocab_size = max(tok_wrapper.vocab_dict.values()) + 1
    if required_vocab_size > base_model.config.vocab_size:
        logger.info(
            f"Resizing embedding table: {base_model.config.vocab_size} -> {required_vocab_size} "
            f"to cover all tokenizer IDs"
        )
        base_model.resize_token_embeddings(required_vocab_size)
        config.vocab_size = required_vocab_size

    model = BiaffineDependencyParser(config)
    model.transformer = base_model
    
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    if "wandb" in training_args.report_to:
        callbacks.append(WandbCallback())
        
    data_collator = DataCollatorForDependencyParsing(tokenizer=tok_wrapper)
    
    trainer = DependencyParsingTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"] if "validation" in tokenized_datasets else tokenized_datasets["test"],
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_parsing_metrics(eval_pred, id2dep_label),
        callbacks=callbacks
    )
    
    train_result = trainer.train()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    main()

# scripts/tokenization/

Training tokenizers, applying them to text, and segmenting corpora.

| File                                  | Purpose                                                        |
| ------------------------------------- | -------------------------------------------------------------- |
| `train_tokenizer.py`                  | Trains one tokenizer. Dispatches to the right backend per family |
| `myte_tokenizer.py`                   | MyTE tokenizer implementation (imported by `train_tokenizer.py`) |
| `apply_tokenizer.py`                  | Tokenizes a single file with a trained model                    |
| `run_segment.py`                      | Segments corpora across a lang × tokenizer grid, uploads to HF   |
| `run_sage_training.py`                | Drives SaGe training (wraps the external `SaGe/` repo)          |
| `extract_and_covert_final_sage_vocab.py` | Converts a finished SaGe run into a usable vocabulary        |
| `run_train.sh` / `run_inference.sh` / `run_segment.sh` / `run_extract_sage.sh` | Shell wrappers; edit the config block at the top of each |

## Training a tokenizer

```bash
python3 scripts/tokenization/train_tokenizer.py \
    --lang eng --model_type bpe --vocab_size 10000 \
    [--input_file PATH] [--output_dir tokenizers-bin] [--character_coverage 1.0]
```

`--model_type` accepts:

| Group                     | Values                                            |
| ------------------------- | ------------------------------------------------- |
| Standard                  | `unigram`, `bpe`, `wordpiece`                      |
| Morphological             | `morfessor`, `myte`                                |
| Recent subword algorithms | `superbpe`, `boundlessbpe`, `pickybpe`, `pathpiece` |
| Morph-presegmented        | `morphbpe`, `morphulm`, `morphwp`                  |
| SSLM-presegmented         | `sslm-bpe`, `sslm-ulm`, `sslm-wp`                  |

The `morph*` and `sslm-*` types read from `presegmented_dataset/` and
`morfessor_presegmented_dataset/` instead of the raw corpus.

SaGe is trained separately via `run_sage_training.py`, since it needs a
multi-stage vocabulary/embedding schedule.

### External dependencies

`superbpe`, `boundlessbpe`, `pickybpe`, `pathpiece`, and `sage` call into the
original authors' repositories, expected as directories at the **repository
root** (`superbpe/`, `boundlessbpe/`, `picky_bpe/`, `pathpiece/`, `SaGe/`).
They are gitignored — clone them yourself. See the root README.

## Input data

With no `--input_file`, the corpus is looked up at `../dataset/{lang}/train.{lang}`
— i.e. a `dataset/` directory **beside** the repository, not inside it.

The corpora are published at
[`SakethVemula/what_tokens_dataset`](https://huggingface.co/datasets/SakethVemula/what_tokens_dataset);
see the root README for the download command.

## Segmentation

```bash
python3 scripts/tokenization/run_segment.py \
    --langs eng hin --model_types bpe unigram --vocab_size 10000 \
    --split test --hf_repo <user>/<dataset-repo> --batch_size 5
```

Writes to `segmented_outputs/` and pushes to the given Hugging Face dataset repo
(`HF_TOKEN` must be exported).

# pretraining/

BERT-style masked language model pretraining on corpora segmented by each
tokenizer. This is the bridge between intrinsic tokenizer quality and downstream
task performance.

## Model scales

| Entrypoint         | Params | Launch scripts   | W&B project                |
| ------------------ | ------ | ---------------- | -------------------------- |
| `pretrain_4M.py`   | ~4M    | `4M_scripts/`    | `bert-pretraining-p3-4M`   |
| `pretrain_8M.py`   | ~8M    | `8M_scripts/`    | `bert-pretraining-p3-8M`   |
| `pretrain_25M.py`  | ~25M   | `25M_scripts/`   | `bert-pretraining-p3-25M`  |

Comparing a metric across these three scales is what produces the scaling
figures (`plots/scale_*.pdf`).

`pretrain.py` and `pretrain_old.py` are earlier iterations kept for provenance;
new runs should use the scale-specific entrypoints above.

## Layout

```
pretrain_{4M,8M,25M}.py   Training entrypoints (HuggingFace Trainer + torchrun)
{4,8,25}M_scripts/        Per-language × per-tokenizer SLURM launch scripts
submit_*.sh               Batch submitters that sweep many configs
pretrain_8M.sh            Generic single-run launcher (takes CLI flags)
plots/                    Loss/perplexity curves and scaling figures
```

## Running

```bash
sbatch pretraining/8M_scripts/eng_morphwp.sh
```

Or configure a single run directly:

```bash
bash pretraining/pretrain_8M.sh \
    --lang eng --model_type bpe --vocab_size 10000
```

Scripts resolve their entrypoint relative to their own location, so they can be
submitted from any directory.

## Arguments

`pretrain_*.py` uses HuggingFace `HfArgumentParser`, so it accepts every
standard `TrainingArguments` flag plus:

| Flag                        | Meaning                                          |
| --------------------------- | ------------------------------------------------ |
| `--language`                | ISO 639-3 code                                    |
| `--model_type`              | Tokenizer family the model is trained on          |
| `--vocab_size`              | Tokenizer vocabulary size                         |
| `--train_data_path`         | Training corpus                                   |
| `--eval_data_path`          | Validation corpus                                 |
| `--tokenizer_dir`           | Where to find tokenizers (default `../tokenizers-bin`) |
| `--bpe_dropout`             | Enable BPE-dropout at training time               |
| `--bpe_dropout_prob`        | Dropout probability (default 0.1)                 |
| `--superbpe_base_vocab_size`| Base vocab for SuperBPE's two-stage training      |

## Before running elsewhere

The `#SBATCH` headers and data paths record the exact cluster setup used for the
paper. Adapt for your environment:

- `--nodelist`, `--mail-user`, and GPU counts in the `#SBATCH` block
- `--train_data_path` / `--eval_data_path` (absolute scratch paths)
- the `source .../bin/activate` virtualenv line
- `WANDB_API_KEY` and `HF_TOKEN` in the environment

## Figures

`plots/` holds training/eval loss and perplexity curves per language, plus the
cross-scale figures. Regenerate with `plots/plot_losses.py` and
`plots/plot_scale.py`.

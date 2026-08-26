# finetuning/

Downstream evaluation of the pretrained models from `pretraining/`. Each
subdirectory is one task, with the same layout:

```
<task>/
  finetune.py   Training/evaluation entrypoint
  run.sh        Single configuration
  run_all.sh    Sweeps tokenizers/vocab sizes for that task
```

## Tasks

| Directory             | Task                          | Metric      |
| --------------------- | ----------------------------- | ----------- |
| `ner_CoNLL/`          | Named entity recognition (CoNLL) | F1       |
| `wiki-ner/`           | Named entity recognition (WikiANN) | F1     |
| `pos/`                | Part-of-speech tagging        | F1          |
| `dependency_parsing/` | Dependency parsing            | LAS         |
| `actsa-te/`           | Telugu sentiment (ACTSA)      | Accuracy    |
| `iitp-pr/`            | Hindi product reviews         | Accuracy    |
| `glue/`               | GLUE benchmark (English)      | Task-specific |

## Running

```bash
bash finetuning/ner_CoNLL/run.sh        # one configuration
bash finetuning/ner_CoNLL/run_all.sh    # sweep the grid for this task
```

Scripts `cd` to their own directory first, so they work from anywhere. On a
cluster, submit with `sbatch` — the `#SBATCH` headers are already in place.

## Arguments

`finetune.py` uses HuggingFace `HfArgumentParser` (all standard
`TrainingArguments` flags apply) plus `--language`, `--dataset_name`,
`--model_type`, `--vocab_size`, and `--tokenizer_dir`.

The task loads the pretrained checkpoint matching `{language}_{model_type}_{vocab_size}`,
so the corresponding `pretraining/` run must have completed first.

## Before running elsewhere

Adapt the `#SBATCH` headers (`--nodelist`, `--mail-user`), the virtualenv
`source` line, and the checkpoint/dataset paths. Export `WANDB_API_KEY` and
`HF_TOKEN` — nothing is hardcoded.

`run_all.sh` scripts clean up `./experiments` and `./wandb` between runs, so
don't keep anything you care about in those paths.

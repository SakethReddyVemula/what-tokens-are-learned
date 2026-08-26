# scripts/

All pipeline code. **Run every command from the repository root** — scripts
resolve data directories relative to the repo root, and the `.sh` wrappers `cd`
there automatically.

| Path             | Purpose                                                     |
| ---------------- | ----------------------------------------------------------- |
| `run_pipeline.py`| Train + evaluate across a whole lang × vocab × type grid      |
| `run_pipeline.sh`| Shell wrapper with the grid configured at the top            |
| `tokenization/`  | Train tokenizers, apply them, segment corpora                |
| `evaluation/`    | Intrinsic metrics and MorphScore                             |
| `analysis/`      | Aggregate raw JSON results, vocabulary overlap               |
| `plotting/`      | Figure generation                                            |
| `slurm/`         | SLURM batch wrappers for the pipeline                        |
| `delete_files_from_hf.py` | Maintenance utility for pruning files from a HF dataset repo |

## The pipeline

```
tokenization/train_tokenizer.py    trains a tokenizer      -> tokenizers-bin/
tokenization/run_segment.py        segments corpora        -> segmented_outputs/
evaluation/evaluate_tokenizers.py  scores segmentations    -> evaluation_results/
analysis/create_csv.py             aggregates JSON -> CSV  -> evaluation_csv_results/
plotting/plot_graphs.py            draws figures           -> plots/
```

`run_pipeline.py` chains the training and evaluation steps for a full grid:

```bash
python3 scripts/run_pipeline.py \
    --langs eng hin tel \
    --vocab_sizes 5000 10000 \
    --model_types bpe unigram \
    [--morphscore --morphscore_data_dir morphscore_data]
```

## Conventions

- **Language codes** are ISO 639-3 (`eng`, `hin`, `tel`, …).
- **Tokenizer artifacts** are named `{lang}_{model_type}_{vocab_size}` inside
  `tokenizers-bin/`.
- **Results** land in `evaluation_results/{metric}/{lang}/{tokenizer}_{vocab}.json`.
- **Credentials** (`HF_TOKEN`, `WANDB_API_KEY`) come from the environment.

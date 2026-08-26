# scripts/evaluation/

Intrinsic evaluation of tokenizers and their segmentations.

| File                        | Purpose                                                       |
| --------------------------- | ------------------------------------------------------------- |
| `evaluate_tokenizers.py`    | Main evaluator — scores a lang × tokenizer grid on chosen metrics |
| `evaluate_tokenizer.py`     | Single-tokenizer evaluation via the external `tokenizer-analysis-suite` |
| `evaluate_sslm_tokenizer.py`| Evaluates SSLM checkpoints (supports sweeping every training step) |
| `evaluate_hnet_tokenizer.py`| Evaluates H-Net tokenizers                                     |
| `run_morphscore.py`         | MorphScore: how well segmentations align with gold morphology   |
| `download_morphscore_data.py`| Fetches MorphScore evaluation datasets                        |
| `run_*.sh`                  | Shell wrappers; edit the config block at the top of each        |

## Metrics

`compression`, `fertility`, `exponence`, `ttr`, `vocab_size`, `length_dist`,
`freq_dist`, `renyi_entropy`, `renyi_efficiency`, `morphscore`

## Running an evaluation

```bash
python3 scripts/evaluation/evaluate_tokenizers.py \
    --langs eng hin tel \
    --model_types bpe unigram \
    --vocab_size 10000 \
    --split test \
    --metrics compression fertility renyi_efficiency \
    --hf_repo <user>/<segments-repo>
```

Segmentations are read from the Hugging Face dataset repo produced by
`scripts/tokenization/run_segment.py`, so export `HF_TOKEN` first.

## Output layout

```
evaluation_results/{metric}/{lang}/{tokenizer}_{vocab_size}.json
```

Each file holds the raw metric values for one tokenizer on one language, e.g.
`evaluation_results/compression/eng/bpe_10000.json`. Aggregate these into CSVs
with `scripts/analysis/create_csv.py`.

## MorphScore

```bash
python3 scripts/evaluation/download_morphscore_data.py --langs eng hin
python3 scripts/evaluation/run_morphscore.py \
    --langs eng hin --model_types bpe --vocab_sizes 10000 --hf_repo <user>/<repo>
```

MorphScore needs the external `morphscore/` repository (with its `data/`
directory) cloned at the repository root; it is gitignored. Results are written
under `evaluation_results/morphscore/` and reported as precision / recall / F1.

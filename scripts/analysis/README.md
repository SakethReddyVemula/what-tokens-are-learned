# scripts/analysis/

Turning raw per-tokenizer JSON results into aggregated tables, plus vocabulary
overlap analysis.

| File                      | Purpose                                                      |
| ------------------------- | ------------------------------------------------------------ |
| `create_csv.py`           | Aggregates `evaluation_results/` JSON into one CSV per metric  |
| `aggregate_metric.py`     | Prints a quick per-language summary of a single metric         |
| `compute_vocab_overlap.py`| Vocabulary overlap between tokenizers, grouped by typology     |
| `run_vocab_overlap.sh`    | Shell wrapper; edit the config block at the top                |

## Aggregating results

```bash
python3 scripts/analysis/create_csv.py --metric fertility
```

Reads `evaluation_results/fertility/**/*.json` and writes
`evaluation_csv_results/fertility/fertility_results.csv`, the tidy table the
plotting scripts consume.

Quick look at one metric without writing files:

```bash
python3 scripts/analysis/aggregate_metric.py --metric fertility --lang eng
```

## Vocabulary overlap

Measures how much vocabulary different tokenizers share for the same language,
optionally grouped by morphological typology (agglutinative, fusional,
analytic/introflexive):

```bash
python3 scripts/analysis/compute_vocab_overlap.py \
    --langs eng hin tel --tokenizers bpe unigram wordpiece \
    --vocab_size 10000 --group_by typology --combined \
    --output_dir plots
```

Useful flags: `--per_lang` (one figure per language), `--compute_stdev`
(variance across languages within a typology group), `--include_sslm` /
`--include_hnet` (add those tokenizer families), `--dry_run` (list what would be
computed), `--no_title` / `--dpi` (figure styling for camera-ready output).

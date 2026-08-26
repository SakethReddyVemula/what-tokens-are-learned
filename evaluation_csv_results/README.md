# evaluation_csv_results/

Aggregated, tidy versions of `evaluation_results/` — one CSV per metric. These
are what the plotting scripts read, and the most convenient entry point if you
want to reuse the paper's numbers.

## Layout

```
evaluation_csv_results/{metric}/{metric}_results.csv
```

Available metrics: `compression`, `exponence`, `fertility`, `length_dist`,
`morphscore`, `morphscore_f1`, `morphscore_precision`, `morphscore_recall`,
`renyi_efficiency`, `ttr`, `vocab_size`.

MorphScore is split into separate precision / recall / F1 tables alongside the
combined one.

## Regenerating

```bash
python3 scripts/analysis/create_csv.py --metric fertility
```

Reads `evaluation_results/fertility/**/*.json` and writes
`fertility/fertility_results.csv`. Re-run per metric after adding new
evaluations.

## Consuming

```bash
python3 scripts/plotting/plot_graphs.py --metric fertility
```

Or load directly with pandas for your own analysis:

```python
import pandas as pd
df = pd.read_csv("evaluation_csv_results/fertility/fertility_results.csv")
```

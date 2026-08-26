# scripts/plotting/

Figure generation for the paper.

| File                      | Purpose                                                        |
| ------------------------- | -------------------------------------------------------------- |
| `plot_graphs.py`          | Main figures: one metric across languages, tokenizers, vocab sizes |
| `plot_metric_dynamics.py` | How a metric evolves over SSLM training checkpoints              |
| `plot_graphs.sh` / `plot_metric_dynamics.sh` | Batch wrappers over many metrics at once      |

Both scripts read the aggregated CSVs from `scripts/analysis/create_csv.py`, so
run that first.

## Main figures

```bash
python3 scripts/plotting/plot_graphs.py \
    --metric fertility \
    --languages eng hin tel --tokenizers bpe unigram --vocab_sizes 5000 10000 \
    --input_dir evaluation_csv_results --output_dir plots \
    [--group_by typology] [--no_legend] [--no_title] [--dpi 300]
```

Every metric at once:

```bash
bash scripts/plotting/plot_graphs.sh
```

## Metric dynamics

Tracks a metric across SSLM training checkpoints rather than final models:

```bash
python3 scripts/plotting/plot_metric_dynamics.py \
    --metric fertility --languages eng hin \
    --input_dir evaluation_results --output_dir plots_dynamics [--pct_steps]
```

## Where figures land

| Directory         | Contents                                          |
| ----------------- | ------------------------------------------------- |
| `plots/`          | Main per-metric figures                            |
| `plots_dynamics/` | Metric-over-training-steps figures                 |
| `plots_combined/` | Multi-panel figures grouped by typology            |
| `plots_stdev/`    | Variance across languages within a typology group  |

Each is written as both `.png` (drafting) and `.pdf` (camera-ready). Use
`--no_title` for figures that get their caption from LaTeX.

# plots_dynamics/

How intrinsic metrics evolve **over SSLM training checkpoints**, rather than at
a single final model. These figures show whether a tokenizer's properties are
stable during training or still drifting.

Metrics tracked: `compression`, `exponence`, `fertility`, `morphscore`,
`mean_token_length`, plus a cross-language MorphScore F1 summary. Each is
written as `.pdf` and `.png`.

## Regenerating

```bash
python3 scripts/plotting/plot_metric_dynamics.py \
    --metric fertility --languages eng hin tel \
    --input_dir evaluation_results --output_dir plots_dynamics [--pct_steps]
```

All metrics at once:

```bash
bash scripts/plotting/plot_metric_dynamics.sh
```

`--pct_steps` plots progress as a percentage of total training steps, which
makes runs of different lengths comparable.

## Source data

Reads `evaluation_results/` directly (not the aggregated CSVs), because it needs
the per-checkpoint granularity that aggregation collapses. Checkpoint results
are named `sslm_checkpoint_{epoch}_{step}_test`.

## Related

- `sslm-morphscores/` — per-language MorphScore CSVs for SSLM checkpoints

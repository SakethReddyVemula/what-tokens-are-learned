# plots/

Main per-metric figures, emitted as both `.png` (drafting) and `.pdf`
(camera-ready).

One figure per intrinsic metric — `compression`, `fertility`, `exponence`,
`ttr`, `vocab_size`, `length_dist`, `renyi_efficiency`, and the MorphScore
family (`morphscore_precision`, `morphscore_recall`, `morphscore_f1`) — each
comparing tokenizers across languages and vocabulary sizes.

## Regenerating

```bash
# Aggregate first, if evaluation_results/ has changed
python3 scripts/analysis/create_csv.py --metric fertility

# Then draw
python3 scripts/plotting/plot_graphs.py --metric fertility --output_dir plots
```

Every metric in one pass:

```bash
bash scripts/plotting/plot_graphs.sh
```

## Related figure directories

| Directory         | Contents                                              |
| ----------------- | ----------------------------------------------------- |
| `plots_dynamics/` | Metrics across SSLM training checkpoints               |
| `plots_combined/` | Multi-panel figures grouped by morphological typology  |
| `plots_stdev/`    | Variance across languages within a typology group      |
| `pretraining/plots/` | LM training curves and cross-scale figures          |

Pass `--no_title` when generating figures whose captions come from LaTeX.

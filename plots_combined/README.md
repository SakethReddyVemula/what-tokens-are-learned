# plots_combined/

Vocabulary overlap figures — how much vocabulary different tokenizers share for
the same language — presented as combined multi-panel views.

```
vocab_overlap.{pdf,png}          Overall overlap across all languages
by_group/                        One panel per group (typology or script)
combined_by_group/               Groups merged into a single multi-panel figure
```

Grouping is by morphological typology (`agglutinative`, `fusional`,
`analytic_introflexive`) or by writing script (`arabic`, `cyrillic`, `latin`, …).

## Regenerating

```bash
python3 scripts/analysis/compute_vocab_overlap.py \
    --langs eng hin tel --tokenizers bpe unigram wordpiece \
    --vocab_size 10000 --group_by typology --combined \
    --output_dir plots_combined
```

See `scripts/analysis/README.md` for the full flag list.

## Related

- `plots_stdev/` — the same analysis reported as variance within each group
- `plots/` — per-metric intrinsic evaluation figures

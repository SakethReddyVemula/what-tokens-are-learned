# plots_stdev/

Vocabulary overlap reported as **variance across languages within a group** —
how consistently a tokenizer behaves for languages that share a typology or
script, rather than the overlap value itself.

```
vocab_overlap.{pdf,png}          Overall
by_group/                        One figure per typology or script group
stdev_by_group/                  Per-group standard deviation tables (Markdown)
```

The `stdev_by_group/*.md` files are the numeric tables behind the figures, one
per typology group (agglutinative, fusional, analytic/introflexive), suitable
for pasting into the paper.

## Regenerating

```bash
python3 scripts/analysis/compute_vocab_overlap.py \
    --langs eng hin tel --tokenizers bpe unigram wordpiece \
    --vocab_size 10000 --group_by typology --compute_stdev \
    --output_dir plots_stdev
```

## Related

- `plots_combined/` — the overlap values themselves, as combined panels
- `scripts/analysis/README.md` — full flag documentation

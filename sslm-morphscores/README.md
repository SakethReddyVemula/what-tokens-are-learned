# sslm-morphscores/

MorphScore results for SSLM tokenizer checkpoints, one CSV per language.

## Layout

```
{lang}_{script}.csv               e.g. eng_latn.csv, fas_arab.csv, hin_deva.csv
morph_f1_all-languages.{pdf,png}  Cross-language F1 summary figure
plot_all_languages_f1.py          Draws the cross-language summary
plot_f1.py                        Single-language F1 curve
plot_cleaner.py                   Restyled variant for camera-ready output
```

Filenames pair the ISO 639-3 language code with its ISO 15924 script code
(`latn`, `arab`, `deva`, `hebr`, `cyrl`, …).

Each CSV tracks MorphScore precision / recall / F1 across SSLM training
checkpoints for that language, which is how morphological alignment is shown to
change as the segmentation model trains.

Files ending in `_old` are superseded figures kept for provenance.

## Regenerating

These CSVs are **exported from a different repository**, not produced here.
`scripts/evaluation/run_morphscore.py` has no SSLM branch — it evaluates fixed
tokenizers only, so it cannot regenerate these files.

The scores, and the underlying per-wordform MorphScore segments, come from
[`transformer-sslm`](https://github.com/SakethReddyVemula/transformer-sslm),
which evaluates each SSLM training checkpoint and writes the
`checkpoint_name,precision,recall,f1` rows collected here.

Given the CSVs, redraw the summary figure:

```bash
python3 sslm-morphscores/plot_all_languages_f1.py
```

## Related

- `plots_dynamics/` — the same checkpoint-level view for other intrinsic metrics
- `scripts/evaluation/README.md` — MorphScore setup and data download

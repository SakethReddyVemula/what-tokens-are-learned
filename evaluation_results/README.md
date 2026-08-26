# evaluation_results/

Raw intrinsic evaluation output — one JSON file per (metric, language,
tokenizer, vocabulary size). This is the unaggregated record behind every number
in the paper.

**This directory is large** (~3.4 GB, ~53,000 files), mostly because SSLM
tokenizers are evaluated at hundreds of training checkpoints each.

## Layout

```
evaluation_results/{metric}/{lang}/{tokenizer}_{vocab_size}.json
```

For example `evaluation_results/compression/eng/bpe_10000.json`:

```json
{
    "compression_rate": 484093,
    "total_subwords": 484093
}
```

## Metrics

| Directory           | Measures                                             |
| ------------------- | ---------------------------------------------------- |
| `compression`       | Corpus size reduction achieved by the tokenizer        |
| `fertility`         | Average subwords produced per word                     |
| `exponence`         | Morphemes expressed per token                          |
| `ttr`               | Type–token ratio                                       |
| `vocab_size`        | Realized vocabulary size                               |
| `length_dist`       | Distribution of token lengths                          |
| `freq_dist`         | Token frequency distribution                           |
| `renyi_entropy`     | Rényi entropy of the token distribution                |
| `renyi_efficiency`  | Rényi efficiency (entropy normalized by vocab size)    |
| `morphscore`        | Agreement with gold morphological boundaries (P/R/F1)  |

## Languages

18 languages (ISO 639-3): `eng fas fin heb hin hrv hun ind kir mal mon rus san
snd swe tam tel tur`

## Regenerating

```bash
python3 scripts/evaluation/evaluate_tokenizers.py \
    --langs eng --model_types bpe --vocab_size 10000 \
    --split test --metrics compression fertility --hf_repo <user>/<repo>
```

## Consuming

Don't read these files directly for analysis — aggregate them first:

```bash
python3 scripts/analysis/create_csv.py --metric fertility
```

That produces the tidy CSVs in `evaluation_csv_results/`, which the plotting
scripts consume.

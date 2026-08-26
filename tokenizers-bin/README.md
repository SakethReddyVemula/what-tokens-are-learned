# tokenizers-bin/

Trained tokenizer artifacts — the models every evaluation and pretraining run
loads.

**Tracked with [Git LFS](https://git-lfs.com)** (~2.1 GB, ~1,700 files). After
cloning:

```bash
git lfs install
git lfs pull
```

Without this you will get small text pointer files instead of real tokenizers,
and loading them will fail.

## Naming

```
{lang}_{model_type}_{vocab_size}.{ext}
```

For example `eng_bpe_10000.model`, `fas_morfessor_10000.bin`,
`hin_myte_10000.json`.

Extensions vary by backend — SentencePiece writes `.model` + `.vocab`,
HuggingFace tokenizers write `.json`, Morfessor and SaGe write `.bin`.

SuperBPE is the exception: because it trains in two stages, it encodes both
vocabulary sizes (`eng_superbpe_4000_10000.json`, i.e. base 4,000 → final
10,000) and also keeps per-stage HuggingFace tokenizer *directories*
(`eng_superbpe_stage1_10000/`, `eng_superbpe_stage2_25000/`) holding
`tokenizer.json`, `merges.txt`, and `vocab.json`.

## Regenerating

```bash
python3 scripts/tokenization/train_tokenizer.py \
    --lang eng --model_type bpe --vocab_size 10000
```

See `scripts/tokenization/README.md` for the full list of supported
`--model_type` values and the external repositories some of them require.

## Note on size

Tokenizers for several families across 18 languages and 4 vocabulary sizes add
up quickly. If you only need a subset, `git lfs pull --include` fetches
selectively:

```bash
git lfs pull --include "tokenizers-bin/eng_*"
```

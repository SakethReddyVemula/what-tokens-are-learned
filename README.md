# What Tokens are Learned when Tokenization is Optimized Jointly with Language Modeling?

> **Accepted to [Findings of EMNLP 2026](https://2026.emnlp.org/).**

Saketh Reddy Vemula, Parameswari Krishnamurthy — IIIT Hyderabad, India
[[arXiv:2608.17325](https://arxiv.org/abs/2608.17325)]

Official code release. Training, evaluation, and analysis code for a
multilingual study of subword tokenizers: how different tokenization algorithms
segment 18 typologically diverse languages, and how those differences propagate
to downstream language model quality.

The suite covers the full experimental pipeline:

1. **Train** tokenizers of several families at matched vocabulary sizes.
2. **Segment** held-out corpora with each tokenizer.
3. **Evaluate** the segmentations on intrinsic metrics (compression, fertility,
   Rényi efficiency, MorphScore, …).
4. **Pretrain** BERT-style models on the resulting segmentations at 4M/8M/12M/25M
   parameter scales.
5. **Finetune** those models on downstream tasks (NER, POS, dependency parsing,
   sentiment, GLUE).
6. **Aggregate and plot** everything into the figures and tables used in the paper.

## Repository layout

```
scripts/
  run_pipeline.py / run_pipeline.sh   Train + evaluate across langs/vocabs in one go
  tokenization/                       Train tokenizers, apply them, segment corpora
  evaluation/                         Intrinsic metrics and MorphScore
  analysis/                           Aggregate raw results, vocabulary overlap
  plotting/                           Figure generation
  slurm/                              SLURM batch wrappers for the pipeline
pretraining/                          BERT pretraining at 4M/8M/12M/25M scales
finetuning/                           Downstream task finetuning (7 tasks)
tokenizers-bin/                       Trained tokenizer binaries (Git LFS)
evaluation_results/                   Raw per-metric JSON results
evaluation_csv_results/               Aggregated CSVs derived from the above
plots/ plots_combined/ plots_dynamics/ plots_stdev/   Generated figures
sslm-morphscores/                     MorphScore results for SSLM checkpoints
```

Each directory has its own `README.md` with details.

> **Run commands from the repository root.** Scripts resolve their data
> directories (`tokenizers-bin/`, `evaluation_results/`, …) relative to the repo
> root, and the shell wrappers `cd` there automatically.

## Experimental grid

**Languages (18, ISO 639-3):** `eng fas fin heb hin hrv hun ind kir mal mon rus
san snd swe tam tel tur`

**Tokenizers:** BPE, BPE-dropout, Unigram, WordPiece, Morfessor, SuperBPE,
BoundlessBPE, PickyBPE, PathPiece, SaGe, MyTE, plus morphologically-informed
(`morphbpe`, `morphulm`, `morphwp`) and SSLM-presegmented (`sslm-bpe`,
`sslm-ulm`, `sslm-wp`) variants.

**Vocabulary sizes:** 5,000 / 10,000 / 20,000 / 30,000 (varies by tokenizer)

**Intrinsic metrics:** `compression`, `fertility`, `exponence`, `ttr`,
`vocab_size`, `length_dist`, `freq_dist`, `renyi_entropy`, `renyi_efficiency`,
`morphscore`

## Setup

Requires Python 3.10+ and [Git LFS](https://git-lfs.com) (tokenizer binaries in
`tokenizers-bin/` are LFS-tracked).

> **Heads-up: a full clone is ~5.5 GB across ~55,000 files.** This is
> deliberate — the repository ships the complete experimental record so every
> number and figure in the paper can be traced back to its source without
> re-running the pipeline:
>
> | Directory              | Size   | What it is |
> | ---------------------- | ------ | ---------- |
> | `evaluation_results/`  | 3.5 GB | Raw per-metric JSON for every language × tokenizer × vocab-size cell (52,957 files) |
> | `tokenizers-bin/`      | 2.1 GB | All 1,718 trained tokenizer binaries, so evaluation is reproducible without retraining |
>
> Everything else — code, figures, aggregated CSVs — is about 60 MB. If you only
> need those, skip the bulk:
>
> ```bash
> # code and figures only; no LFS binaries, shallow history
> GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
>     https://github.com/SakethReddyVemula/what-tokens-are-learned.git
> ```
>
> The aggregated CSVs behind the paper's tables and plots live in
> `evaluation_csv_results/` (~140 KB) — for most readers those are enough, and
> the raw JSON is only needed to recompute them.

For the full artifact:

```bash
git lfs install
git clone https://github.com/SakethReddyVemula/what-tokens-are-learned.git
cd what-tokens-are-learned
git lfs pull
```

Core dependencies: `sentencepiece`, `tokenizers`, `transformers`, `datasets`,
`huggingface_hub`, `morfessor`, `numpy`, `pandas`, `matplotlib`, `seaborn`.

### Companion repositories

The two tokenizer-free models in the paper are trained in separate forks, not in
this repository. Both are required to reproduce the SSLM and H-Net results:

| Repository | Role |
| ---------- | ---- |
| [`transformer-sslm`](https://github.com/SakethReddyVemula/transformer-sslm) | SSLM training and segmentation. Also holds the **SSLM MorphScore segments** — `run_morphscore.py` here has no SSLM branch |
| [`hnet-impl`](https://github.com/SakethReddyVemula/hnet-impl/tree/e2e-branch) (branch `e2e-branch`) | H-Net training and segmentation. Use the `e2e-branch` branch, not `main` |

The `sslm-*` tokenizers in `tokenizers-bin/` are trained here from corpora that
`transformer-sslm` pre-segments; see `scripts/tokenization/README.md`. The
per-checkpoint MorphScore scores in `sslm-morphscores/` are aggregates exported
from that repo.

### External tokenizer implementations

Several tokenizers are trained by calling into their original authors'
repositories, which are **not vendored here**. Clone them into the repository
root before training those families:

| Directory       | Needed for               |
| --------------- | ------------------------ |
| `superbpe/`     | SuperBPE                 |
| `boundlessbpe/` | BoundlessBPE             |
| `picky_bpe/`    | PickyBPE                 |
| `pathpiece/`    | PathPiece                |
| `SaGe/`         | SaGe                     |
| `morphscore/`   | MorphScore evaluation    |

These are listed in `.gitignore` so they never enter this repository's history.

### Credentials

Scripts read credentials from the environment — nothing is hardcoded:

```bash
export HF_TOKEN="..."        # Hugging Face Hub (segment/result uploads)
export WANDB_API_KEY="..."   # Weights & Biases (pretraining/finetuning logs)
```

### Dataset

The training corpora are published at
[`SakethVemula/what_tokens_dataset`](https://huggingface.co/datasets/SakethVemula/what_tokens_dataset):

```bash
# from the directory *containing* this repository
huggingface-cli download SakethVemula/what_tokens_dataset \
    --repo-type dataset --local-dir dataset
```

The dataset covers the paper's 18 languages plus a few additional ones
(`afr`, `ell`, `gle`, `isl`, `ita`, `kat`, `kor`, `lav`, `spa`); pass
`--include "{lang}/*"` to fetch only the subset you need.

Pre-segmented corpora and all model checkpoints are released separately — see
[Released artifacts](#released-artifacts).

Scripts expect the result in a `dataset/` directory **beside** the repo (not
inside it), laid out as:

```
dataset/
  {lang}/
    train.{lang}
    valid.{lang}
    test.{lang}
  presegmented_dataset/           # for sslm-* model types
  morfessor_presegmented_dataset/ # for morph* model types
```

## Quick start

Train one tokenizer and evaluate it:

```bash
python3 scripts/tokenization/train_tokenizer.py --lang eng --model_type bpe --vocab_size 10000
python3 scripts/evaluation/evaluate_tokenizer.py --lang eng --model_type bpe --vocab_size 10000
```

Sweep the grid:

```bash
python3 scripts/run_pipeline.py \
    --langs eng hin tel --vocab_sizes 5000 10000 --model_types bpe unigram
```

Aggregate results and draw figures:

```bash
python3 scripts/analysis/create_csv.py --metric fertility
python3 scripts/plotting/plot_graphs.py --metric fertility
```

## Reproducing the paper

```bash
# 1. Train tokenizers across the grid
bash scripts/tokenization/run_train.sh

# 2. Segment held-out corpora
bash scripts/tokenization/run_segment.sh

# 3. Intrinsic evaluation
bash scripts/evaluation/run_evaluation.sh
bash scripts/evaluation/run_morphscore.sh

# 4. Pretrain LMs on the segmentations   (see pretraining/README.md)
sbatch pretraining/8M_scripts/eng_morphwp.sh

# 5. Downstream finetuning               (see finetuning/README.md)
bash finetuning/ner_CoNLL/run_all.sh

# 6. Aggregate + plot
python3 scripts/analysis/create_csv.py --metric <metric>
bash scripts/plotting/plot_graphs.sh
```

Steps 1–3 depend only on the corpora; steps 4–5 require GPUs and were run on a
SLURM cluster (see `scripts/slurm/` and the `#SBATCH` headers in the
pretraining/finetuning scripts, which encode the exact resources used).

## Released artifacts

All datasets and model checkpoints for the paper are published on the Hugging
Face Hub under [`SakethVemula`](https://huggingface.co/SakethVemula).

### Corpora

| Resource | Description |
| -------- | ----------- |
| [`what_tokens_dataset`](https://huggingface.co/datasets/SakethVemula/what_tokens_dataset) | Non-parallel monolingual corpora used to train the tokenizers, tokenizer-free LMs, and BERT models |

### Pretrained models

| Resource | Description |
| -------- | ----------- |
| [`sslm-models`](https://huggingface.co/SakethVemula/sslm-models) | SSLM checkpoints |
| [`hnet-models`](https://huggingface.co/SakethVemula/hnet-models) | H-Net checkpoints |
| [`BERT-models-8M`](https://huggingface.co/SakethVemula/BERT-models-8M) | BERT models across tokenizers (main comparison, 8M params) |
| [`BERT-models-4M`](https://huggingface.co/SakethVemula/BERT-models-4M) | BERT models, 4M params (scalability analysis) |
| [`BERT-models-25M`](https://huggingface.co/SakethVemula/BERT-models-25M) | BERT models, 25M params (scalability analysis) |

### Pre-segmented corpora

| Resource | Segmented by |
| -------- | ------------ |
| [`fixed-tokenizer-segments`](https://huggingface.co/datasets/SakethVemula/fixed-tokenizer-segments) | Fixed tokenizers |
| [`sslm-corpus-segments`](https://huggingface.co/datasets/SakethVemula/sslm-corpus-segments) | SSLMs |
| [`hnet-segments`](https://huggingface.co/datasets/SakethVemula/hnet-segments) | H-Nets |

### Pre-segmented MorphScore data

| Resource | Segmented by |
| -------- | ------------ |
| [`fixed-tokenizer-morphscore-segments`](https://huggingface.co/datasets/SakethVemula/fixed-tokenizer-morphscore-segments) | Fixed tokenizers |
| [`transformer-sslm`](https://github.com/SakethReddyVemula/transformer-sslm) (in-repo, not on the Hub) | SSLMs |
| [`hnet-morphscore-segments`](https://huggingface.co/datasets/SakethVemula/hnet-morphscore-segments) | H-Nets |

## Citation

Citation of the Findings of EMNLP 2026 version TBA once it appears in the ACL
Anthology. Until then, the arXiv preprint:

```bibtex
@misc{vemula2026tokenslearnedtokenizationoptimized,
      title={What Tokens are Learned when Tokenization is Optimized Jointly with Language Modeling?}, 
      author={Saketh Reddy Vemula and Parameswari Krishnamurthy},
      year={2026},
      eprint={2608.17325},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2608.17325}, 
}
```

#!/usr/bin/env python3
"""
compute_vocab_overlap.py
------------------------
Computes pairwise Jaccard vocabulary overlap between different tokenizers
based on their *segmented* corpus output (not the static tokenizer vocabulary).

For each requested tokenizer and language the script:
  1. Downloads the segmented JSONL/JSON file from the appropriate HF repo.
  2. Extracts the set of unique token strings (the "segmented vocabulary").
  3. Averages the per-language Jaccard similarity matrices into one overall matrix.
  4. Plots a lower-triangular heatmap with a 0–1 colour scale.

Jaccard similarity: |vocab(A) ∩ vocab(B)| / |vocab(A) ∪ vocab(B)|

Supported tokenizer sources
---------------------------
  fixed   : SakethVemula/fixed-tokenizer-segments  (JSONL, list-of-lists per line)
  sslm    : SakethVemula/sslm-corpus-segments       (JSONL, list-of-lists per line)
  hnet    : SakethVemula/hnet-segments              (JSON array)

Usage
-----
python3 compute_vocab_overlap.py \\
    --langs fin hun mal tam tel kir tur mon ind san hin snd hrv rus fas eng swe heb \\
    --tokenizers bpe unigram wordpiece morfessor \\
    --vocab_size 25000 \\
    --include_sslm \\
    --include_hnet \\
    --output_dir plots
"""

import argparse
import json
import os
import re
import shutil
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from huggingface_hub import hf_hub_download, HfApi

# ── language group mappings (mirrored from plot_graphs.py) ───────────────────
LANG_GROUPS = {
    "fin": "Agglutinative", "hun": "Agglutinative", "mal": "Agglutinative",
    "tam": "Agglutinative", "tel": "Agglutinative", "kir": "Agglutinative",
    "tur": "Agglutinative", "mon": "Agglutinative", "ind": "Agglutinative",
    "san": "Fusional",      "hin": "Fusional",      "snd": "Fusional",
    "hrv": "Fusional",      "rus": "Fusional",      "fas": "Fusional",
    "eng": "Analytic & Introflexive", "swe": "Analytic & Introflexive",
    "heb": "Analytic & Introflexive",
}

# ── broad typology groups for combined plots (2 groups) ───────────────────────
# Agglutinative stays as-is; Fusional + Analytic & Introflexive are merged.
BROAD_LANG_GROUPS = {
    "fin": "Agglutinative", "hun": "Agglutinative", "mal": "Agglutinative",
    "tam": "Agglutinative", "tel": "Agglutinative", "kir": "Agglutinative",
    "tur": "Agglutinative", "mon": "Agglutinative", "ind": "Agglutinative",
    "san": "Fusional & Analytic", "hin": "Fusional & Analytic", "snd": "Fusional & Analytic",
    "hrv": "Fusional & Analytic", "rus": "Fusional & Analytic", "fas": "Fusional & Analytic",
    "eng": "Fusional & Analytic", "swe": "Fusional & Analytic", "heb": "Fusional & Analytic",
}

LANG_SCRIPTS = {
    "fin": "Latin",      "hun": "Latin",      "tur": "Latin",
    "ind": "Latin",      "hrv": "Latin",      "eng": "Latin",      "swe": "Latin",
    "kir": "Cyrillic",   "mon": "Cyrillic",   "rus": "Cyrillic",
    "san": "Devanagari", "hin": "Devanagari",
    "snd": "Arabic",     "fas": "Arabic",
    "mal": "Malayalam",
    "tam": "Tamil",
    "tel": "Telugu",
    "heb": "Hebrew",
}

# ── tokenizer display names (shared with plot_graphs.py) ──────────────────────
TOKENIZER_MAP = {
    "bpe-dropout":       "BPE-Dropout",
    "bpe":               "BPE",
    "hnet":              "H-Nets",
    "morphbpe":          "MorphBPE",
    "morphbpe-dropout":  "MorphBPE-Dropout",
    "morphulm":          "MorphULM",
    "morphwp":           "MorphWP",
    "myte":              "MYTE",
    "boundlessbpe":      "BoundlessBPE",
    "superbpe":          "SuperBPE",
    "morfessor":         "Morfessor",
    "unigram":           "Unigram",
    "wordpiece":         "WordPiece",
    "pathpiece":         "Pathpiece",
    "pickybpe":          "PickyBPE",
    "sage":              "SaGe",
    "sslm":              "SSLM",
}


def display_name(tok_key: str) -> str:
    return TOKENIZER_MAP.get(tok_key.lower(), tok_key)


# ── vocabulary extraction helpers ──────────────────────────────────────────────

def extract_vocab_fixed(file_path: str, is_superbpe: bool = False) -> set:
    """Extract unique tokens from a fixed-tokenizer JSONL file.

    Each line is a JSON array of word-level token lists:
        [["bur", "ied"], ["and"], ...]
    """
    vocab = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                words = json.loads(line)
            except json.JSONDecodeError:
                continue
            for word_tokens in words:
                for tok in word_tokens:
                    vocab.add(tok)
    return vocab


def extract_vocab_sslm(file_path: str) -> set:
    """Extract unique tokens from an SSLM JSONL segmentation file.

    Format is identical to fixed-tokenizer JSONL.
    """
    return extract_vocab_fixed(file_path)


def extract_vocab_hnet(file_path: str) -> set:
    """Extract unique tokens from an H-Net JSON segmentation file.

    Top-level structure: list of sentence groups, each a list of token lists.
    The special start token ``\\xfe`` is filtered out.
    """
    vocab = set()
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  [warn] JSON parse error in {file_path}: {e}")
            return vocab

    for sentence_group in data:
        if not sentence_group:
            continue
        for word_tokens in sentence_group:
            for tok in word_tokens:
                if tok != "\\xfe":
                    vocab.add(tok)
    return vocab


# ── Jaccard ────────────────────────────────────────────────────────────────────

def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0          # both empty → fully "overlap"
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


# ── HF download helpers ────────────────────────────────────────────────────────

def _download(repo_id: str, path_in_repo: str, cache_dir: str):
    """Download a file from a HF dataset repo. Returns local path or None."""
    try:
        return hf_hub_download(
            repo_id=repo_id,
            filename=path_in_repo,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
    except Exception as e:
        print(f"  [warn] Cannot download '{path_in_repo}' from '{repo_id}': {e}")
        return None


# ── collect vocabularies ───────────────────────────────────────────────────────

def collect_fixed_vocab(
    lang: str,
    model_type: str,
    vocab_size: int,
    split: str,
    repo: str,
    bpe_dropout: bool,
    bpe_dropout_prob: float,
    superbpe_base_vocab_size: int,
    cache_dir: str,
) -> set | None:
    """Download and extract vocab for one fixed-tokenizer × language pair."""
    output_model_type = model_type
    if bpe_dropout and model_type == "bpe":
        output_model_type = "bpe-dropout"
    elif bpe_dropout and model_type == "morphbpe":
        output_model_type = "morphbpe-dropout"

    if model_type == "superbpe":
        filename = f"{output_model_type}_v{superbpe_base_vocab_size}t-{vocab_size}T_{split}.jsonl"
    else:
        filename = f"{output_model_type}_v{vocab_size}_{split}.jsonl"

    path_in_repo = f"{lang}/{filename}"
    local = _download(repo, path_in_repo, cache_dir)
    if local is None:
        return None
    return extract_vocab_fixed(local, is_superbpe=(model_type == "superbpe"))


def collect_sslm_vocab(
    lang: str,
    repo: str,
    cache_dir: str,
) -> set | None:
    """Download and extract vocab for SSLM × language (highest checkpoint)."""
    api = HfApi()
    try:
        all_files = list(api.list_repo_files(repo_id=repo, repo_type="dataset"))
    except Exception as e:
        print(f"  [warn] Cannot list files in '{repo}': {e}")
        return None

    lang_files = [
        f for f in all_files
        if f.startswith(f"{lang}/") and f.endswith(".jsonl")
    ]

    # Find highest checkpoint
    candidates = []
    for fpath in lang_files:
        fname = os.path.basename(fpath)
        m = re.search(r"checkpoint_?([\d_]+)", fname)
        if m:
            candidates.append((fpath, m.group(1)))

    if not candidates:
        print(f"  [warn] No SSLM checkpoint files found for '{lang}'.")
        return None

    def parse_step(s):
        parts = s.split("_")
        if len(parts) == 1:
            return (int(parts[0]), 0)
        elif len(parts) == 2:
            return (int(parts[0]), int(parts[1]))
        return (0, 0)

    best_path, best_step = max(candidates, key=lambda x: parse_step(x[1]))
    print(f"    SSLM: using checkpoint {best_step} for {lang}")
    local = _download(repo, best_path, cache_dir)
    if local is None:
        return None
    return extract_vocab_sslm(local)


def collect_hnet_vocab(lang: str, repo: str, cache_dir: str) -> set | None:
    """Download and extract vocab for H-Net × language."""
    filename = f"seg_{lang}_best_model.pt.json"
    path_in_repo = f"{lang}/{filename}"
    local = _download(repo, path_in_repo, cache_dir)
    if local is None:
        return None
    return extract_vocab_hnet(local)


# ── matrix / label filtering ─────────────────────────────────────────────────

def filter_by_valid(
    matrix: np.ndarray,
    labels: list,
    valid_mask: list,
) -> tuple:
    """Return (sub_matrix, sub_labels) keeping only rows/cols where valid_mask is True."""
    idx = [i for i, v in enumerate(valid_mask) if v]
    if not idx:
        return np.zeros((0, 0)), []
    sub = matrix[np.ix_(idx, idx)]
    sub_labels = [labels[i] for i in idx]
    return sub, sub_labels


# ── plotting ───────────────────────────────────────────────────────────────────

def plot_heatmap(
    matrix: np.ndarray,
    labels: list,
    output_dir: str,
    filename_stem: str = "vocab_overlap",
    title: str = "Pairwise Vocabulary Overlap (Jaccard)",
    dpi: int = 200,
):
    """Render a lower-triangular heatmap of pairwise Jaccard similarities.

    Args:
        matrix : N×N numpy array, values in [0, 1].
        labels : list of N display-name strings.
        output_dir : directory to save PDF and PNG.
    """
    n = len(labels)
    # Mask upper triangle (above diagonal)
    mask = np.zeros_like(matrix, dtype=bool)
    mask[np.triu_indices(n, k=1)] = True
    masked = np.ma.array(matrix, mask=mask)

    # Figure sizing: give roughly 0.55 in per cell
    cell = max(0.50, 6.0 / n)
    fig_size = max(5.0, n * cell + 2.0)
    fig, ax = plt.subplots(figsize=(fig_size + 1.5, fig_size))

    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color="white")

    im = ax.imshow(
        masked,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
        interpolation="nearest",
    )

    # ── colourbar ──
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Jaccard overlap", fontsize=11)
    cbar.set_ticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # ── cell annotations ──
    for i in range(n):
        for j in range(n):
            if j > i:
                continue           # skip upper triangle
            val = matrix[i, j]
            text_color = "white" if (val < 0.25 or val > 0.75) else "black"
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                fontsize=max(10, 9 - n // 4),
                color=text_color,
                fontweight="bold" if i == j else "normal",
            )

    # ── bold border on diagonal squares ──
    for k in range(n):
        rect = mpatches.FancyBboxPatch(
            (k - 0.5, k - 0.5), 1.0, 1.0,
            boxstyle="square,pad=0",
            linewidth=2.0,
            edgecolor="black",
            facecolor="none",
            zorder=5,
        )
        ax.add_patch(rect)

    # ── axes ──
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=max(10, 10 - n // 4))
    ax.set_yticklabels(labels, fontsize=max(10, 10 - n // 4))

    # Move x-ticks to bottom (default) and y-ticks to left (default)
    ax.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True, labeltop=False)

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=14)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        out_path = os.path.join(output_dir, f"{filename_stem}.{ext}")
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved → {out_path}")

    plt.close(fig)


def plot_combined_heatmap(
    mean_mat: np.ndarray,
    stdev_mat: np.ndarray,
    labels: list,
    output_dir: str,
    filename_stem: str,
    title: str = "",
    dpi: int = 200,
    vmax_stdev: float | None = None,
):
    """Render a combined heatmap:
      - lower-left triangle  : mean Jaccard overlap  (RdBu_r, 0–1)
      - upper-right triangle : std-dev of Jaccard    (YlOrRd, auto scale)
      - diagonal             : white / empty

    Two colorbars are placed in manually-positioned axes so they never
    overlap each other or the heatmap.
    """
    n = len(labels)

    # ── colourmaps — masked cells transparent so layers don't occlude each other
    cmap_overlap = plt.cm.RdBu_r.copy()
    cmap_overlap.set_bad(alpha=0)
    cmap_stdev = plt.cm.YlOrRd.copy()
    cmap_stdev.set_bad(alpha=0)

    # ── build masked arrays ──
    # Overlap: show only lower triangle (i >= j); mask upper + diagonal for this layer
    mask_lo = np.zeros((n, n), dtype=bool)
    mask_lo[np.triu_indices(n, k=0)] = True          # mask diagonal + upper
    masked_lo = np.ma.array(mean_mat, mask=mask_lo)

    # Stdev: show only upper triangle (i < j); mask lower + diagonal for this layer
    valid_stdev = np.nan_to_num(stdev_mat, nan=0.0)
    mask_hi = np.zeros((n, n), dtype=bool)
    mask_hi[np.tril_indices(n, k=0)] = True          # mask diagonal + lower
    mask_hi |= np.isnan(stdev_mat)                   # also mask NaN cells
    masked_hi = np.ma.array(valid_stdev, mask=mask_hi)

    valid_svals = stdev_mat[~np.isnan(stdev_mat)]
    if vmax_stdev is None:
        vmax_stdev = float(np.max(valid_svals)) if valid_svals.size > 0 else 1.0
    vmax_stdev = max(vmax_stdev, 1e-6)

    # ── figure & axes ──
    cell = max(0.50, 6.0 / n)
    fig_size = max(5.0, n * cell + 2.0)
    fig, ax = plt.subplots(figsize=(fig_size + 2.0, fig_size))

    # Pre-carve right margin for colorbars BEFORE drawing
    fig.subplots_adjust(right=0.78)
    ax.set_facecolor("white")   # transparent masked cells (diagonal) → white

    # ── draw overlap layer (lower-left triangle) ──
    im_lo = ax.imshow(
        masked_lo,
        cmap=cmap_overlap,
        vmin=0.0, vmax=1.0,
        aspect="equal",
        interpolation="nearest",
    )

    # ── draw stdev layer (upper-right triangle, on top) ──
    im_hi = ax.imshow(
        masked_hi,
        cmap=cmap_stdev,
        vmin=0.0, vmax=vmax_stdev,
        aspect="equal",
        interpolation="nearest",
    )

    # ── cell annotations ──
    for i in range(n):
        for j in range(n):
            if i == j:
                continue   # diagonal: skip
            if j < i:
                # lower triangle: overlap
                val = mean_mat[i, j]
                text_color = "white" if (val < 0.25 or val > 0.75) else "black"
                ax.text(j, i, f"{val:.2f}",
                        ha="center", va="center",
                        fontsize=max(7, 9 - n // 4),
                        color=text_color)
            else:
                # upper triangle: stdev
                if np.isnan(stdev_mat[i, j]):
                    continue
                val = stdev_mat[i, j]
                rel = val / vmax_stdev if vmax_stdev > 0 else 0.0
                text_color = "white" if rel > 0.65 else "black"
                ax.text(j, i, f"{val:.3f}",
                        ha="center", va="center",
                        fontsize=max(7, 9 - n // 4),
                        color=text_color)

    # ── bold diagonal borders ──
    for k in range(n):
        rect = mpatches.FancyBboxPatch(
            (k - 0.5, k - 0.5), 1.0, 1.0,
            boxstyle="square,pad=0",
            linewidth=2.0, edgecolor="black", facecolor="white", zorder=5,
        )
        ax.add_patch(rect)

    # ── axes labels ──
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=max(10, 11 - n // 4))
    ax.set_yticklabels(labels, fontsize=max(10, 11 - n // 4))
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=14)

    # ── colorbars in manually-positioned axes so they never overlap ──
    fig.canvas.draw()           # commit layout so ax.get_position() is accurate
    pos = ax.get_position()     # Bbox in figure-fraction coords

    gap    = 0.06              # gap between the two bars
    cbar_w = 0.018              # narrower bars
    x0     = pos.x1 + 0.020    # start just right of the heatmap
    mid_y  = pos.y0 + pos.height / 2.0
    # Each bar occupies 38 % of the heatmap height (shorter than the full half)
    bar_h  = pos.height * 0.38
    top_y  = mid_y + gap / 2.0
    bot_y  = mid_y - gap / 2.0 - bar_h

    # Bottom bar → overlap (RdBu_r)
    cax_lo = fig.add_axes([x0, bot_y, cbar_w, bar_h])
    cbar_lo = fig.colorbar(im_lo, cax=cax_lo)
    cbar_lo.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar_lo.ax.tick_params(labelsize=7)
    cax_lo.set_title("Mean Jaccard\noverlap",
                     fontsize=7, pad=3, loc="center")

    # Top bar → stdev (YlOrRd)
    cax_hi = fig.add_axes([x0, top_y, cbar_w, bar_h])
    cbar_hi = fig.colorbar(im_hi, cax=cax_hi)
    cbar_hi.ax.tick_params(labelsize=7)
    cax_hi.set_title("Std-dev of\nJaccard",
                     fontsize=7, pad=3, loc="center")

    os.makedirs(output_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        out_path = os.path.join(output_dir, f"{filename_stem}.{ext}")
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved → {out_path}")

    plt.close(fig)


def plot_stdev_heatmap(
    stdev_matrix: np.ndarray,
    labels: list,
    output_dir: str,
    filename_stem: str,
    title: str = "",
    dpi: int = 200,
):
    """Render a lower-triangular heatmap of per-cell Jaccard std-dev across languages.

    Uses a sequential colormap (viridis) so that higher std-dev (more variation)
    is visually prominent.  Values are in [0, 1] but in practice stay well below
    0.5, so the colorbar auto-scales to the data's max for readability.

    Args:
        stdev_matrix : N×N numpy array of std-dev values (NaN where < 2 languages).
        labels       : list of N display-name strings.
        output_dir   : directory to save PDF and PNG.
        filename_stem: base filename (no extension).
        title        : plot title (empty string → no title).
        dpi          : output resolution.
    """
    n = len(labels)

    # Mask upper triangle and cells with no valid data (NaN)
    mask = np.zeros_like(stdev_matrix, dtype=bool)
    mask[np.triu_indices(n, k=1)] = True
    mask |= np.isnan(stdev_matrix)
    masked = np.ma.array(np.nan_to_num(stdev_matrix, nan=0.0), mask=mask)

    # Determine colour scale from actual data range
    valid_vals = stdev_matrix[~np.isnan(stdev_matrix)]
    vmax = float(np.max(valid_vals)) if valid_vals.size > 0 else 1.0
    vmax = max(vmax, 1e-6)   # avoid degenerate all-zero scale

    cell = max(0.50, 6.0 / n)
    fig_size = max(5.0, n * cell + 2.0)
    fig, ax = plt.subplots(figsize=(fig_size + 1.5, fig_size))

    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="white")

    im = ax.imshow(
        masked,
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        aspect="equal",
        interpolation="nearest",
    )

    # ── colourbar ──
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Std-dev of Jaccard overlap", fontsize=11)

    # ── cell annotations ──
    for i in range(n):
        for j in range(n):
            if j > i or mask[i, j]:
                continue
            val = stdev_matrix[i, j]
            # Use white text when the cell is dark (high relative value)
            rel = val / vmax if vmax > 0 else 0.0
            text_color = "white" if rel > 0.65 else "black"
            ax.text(
                j, i, f"{val:.3f}",
                ha="center", va="center",
                fontsize=max(8, 9 - n // 4),
                color=text_color,
                fontweight="bold" if i == j else "normal",
            )

    # ── bold border on diagonal squares ──
    for k in range(n):
        rect = mpatches.FancyBboxPatch(
            (k - 0.5, k - 0.5), 1.0, 1.0,
            boxstyle="square,pad=0",
            linewidth=2.0,
            edgecolor="black",
            facecolor="none",
            zorder=5,
        )
        ax.add_patch(rect)

    # ── axes ──
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=max(10, 10 - n // 4))
    ax.set_yticklabels(labels, fontsize=max(10, 10 - n // 4))
    ax.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=14)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        out_path = os.path.join(output_dir, f"{filename_stem}.{ext}")
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved → {out_path}")

    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute pairwise vocabulary overlap between tokenizer segmented outputs "
            "and plot a lower-triangular Jaccard heatmap."
        )
    )

    # Languages & tokenizers
    parser.add_argument(
        "--langs", nargs="+",
        default="fin hun mal tam tel kir tur mon ind san hin snd hrv rus fas eng swe heb".split(),
        help="Language codes to include.",
    )
    parser.add_argument(
        "--tokenizers", nargs="+",
        default=["bpe", "unigram", "wordpiece", "morfessor", "superbpe",
                 "boundlessbpe", "pathpiece", "myte", "sage",
                 "morphbpe", "morphulm", "morphwp"],
        help="Fixed tokenizer types to include.",
    )
    parser.add_argument("--vocab_size", type=int, default=25000)
    parser.add_argument("--split", type=str, default="test")

    # BPE dropout
    parser.add_argument("--bpe_dropout", action="store_true")
    parser.add_argument("--bpe_dropout_prob", type=float, default=0.1)
    parser.add_argument("--superbpe_base_vocab_size", type=int, default=10000)

    # HF repos
    parser.add_argument("--fixed_repo", default="SakethVemula/fixed-tokenizer-segments")
    parser.add_argument("--sslm_repo", default="SakethVemula/sslm-corpus-segments")
    parser.add_argument("--hnet_repo", default="SakethVemula/hnet-segments")

    # Opt-in for neural models
    parser.add_argument("--include_sslm", action="store_true",
                        help="Include SSLM in the overlap comparison.")
    parser.add_argument("--include_hnet", action="store_true",
                        help="Include H-Nets in the overlap comparison.")

    # Output / aggregation
    parser.add_argument("--output_dir", default="plots")
    parser.add_argument("--per_lang", action="store_true",
                        help="Produce one heatmap per language in addition to the aggregate.")
    parser.add_argument("--group_by", default="typology", choices=["typology", "script", "both"],
                        help="Group languages for group-averaged plots: 'typology' (default), 'script', or 'both'.")
    parser.add_argument("--compute_stdev", action="store_true",
                        help="Also plot std-dev heatmaps showing variation of Jaccard overlap across "
                             "languages within each typology/script group.")
    parser.add_argument("--combined", action="store_true",
                        help="Plot a combined heatmap per broad typology group (Agglutinative vs "
                             "Fusional & Analytic): lower-left = mean Jaccard overlap, "
                             "upper-right = std-dev, diagonal = empty.")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--no_title", action="store_true",
                        help="Omit the plot title.")

    # Dry-run (synthetic data for smoke-testing without HF access)
    parser.add_argument("--dry_run", action="store_true",
                        help="Use synthetic random vocabularies instead of downloading from HF.")

    args = parser.parse_args()

    # ── build list of (tok_key, source) tuples ──
    entries = []
    for tok in args.tokenizers:
        entries.append((tok, "fixed"))
    if args.include_sslm:
        entries.append(("sslm", "sslm"))
    if args.include_hnet:
        entries.append(("hnet", "hnet"))

    if not entries:
        sys.exit("No tokenizers specified. Pass at least one via --tokenizers or --include_sslm / --include_hnet.")

    labels = [display_name(k) for k, _ in entries]
    n = len(entries)

    print(f"Tokenizers ({n}): {labels}")
    print(f"Languages  ({len(args.langs)}): {args.langs}")
    print()

    # ── accumulate Jaccard matrices across languages ──
    sum_matrix = np.zeros((n, n), dtype=float)
    count_matrix = np.zeros((n, n), dtype=int)

    cache_fixed = "temp_cache_overlap_fixed"
    cache_sslm  = "temp_cache_overlap_sslm"
    cache_hnet  = "temp_cache_overlap_hnet"

    # Per-language storage if requested
    lang_matrices = {}  # lang -> np.ndarray

    for lang in args.langs:
        print(f"\n{'='*60}")
        print(f"Language: {lang}")
        print(f"{'='*60}")

        # Collect vocab for each tokenizer
        vocabs = []
        for tok_key, source in entries:
            if args.dry_run:
                # Synthetic: random set of strings for smoke testing
                rng = np.random.default_rng(abs(hash((lang, tok_key))) % (2**32))
                size = int(rng.integers(500, 3000))
                chars = "abcdefghijklmnopqrstuvwxyz"
                tok_set = {
                    "".join(rng.choice(list(chars), size=int(rng.integers(2, 8))))
                    for _ in range(size)
                }
                print(f"  [dry_run] {tok_key}: {len(tok_set)} synthetic tokens")
                vocabs.append(tok_set)
                continue

            print(f"  Downloading: {tok_key} ({source})")
            if source == "fixed":
                vocab = collect_fixed_vocab(
                    lang=lang,
                    model_type=tok_key,
                    vocab_size=args.vocab_size,
                    split=args.split,
                    repo=args.fixed_repo,
                    bpe_dropout=args.bpe_dropout,
                    bpe_dropout_prob=args.bpe_dropout_prob,
                    superbpe_base_vocab_size=args.superbpe_base_vocab_size,
                    cache_dir=cache_fixed,
                )
            elif source == "sslm":
                vocab = collect_sslm_vocab(lang=lang, repo=args.sslm_repo, cache_dir=cache_sslm)
            elif source == "hnet":
                vocab = collect_hnet_vocab(lang=lang, repo=args.hnet_repo, cache_dir=cache_hnet)
            else:
                vocab = None

            if vocab is None:
                print(f"    → skipped (not available)")
            else:
                print(f"    → {len(vocab):,} unique tokens")
            vocabs.append(vocab)

        # ── compute per-lang Jaccard matrix ──
        lang_mat = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(n):
                va, vb = vocabs[i], vocabs[j]
                if va is None or vb is None:
                    continue
                lang_mat[i, j] = jaccard(va, vb)

        lang_matrices[lang] = lang_mat

        # Accumulate for aggregate
        valid = ~np.isnan(lang_mat)
        sum_matrix[valid] += lang_mat[valid]
        count_matrix[valid] += 1

        # Optionally save per-language plot
        if args.per_lang:
            lang_dir = os.path.join(args.output_dir, "per_lang")
            # Only include tokenizers that had data for THIS language
            lang_valid = [v is not None for v in vocabs]
            skipped = [labels[i] for i, v in enumerate(lang_valid) if not v]
            if skipped:
                print(f"  [per_lang:{lang}] Dropping from plot (no data): {skipped}")
            nonnan = lang_mat.copy()
            nonnan[np.isnan(nonnan)] = 0.0
            sub_mat, sub_labels = filter_by_valid(nonnan, labels, lang_valid)
            if sub_mat.size == 0:
                print(f"  [per_lang:{lang}] No tokenizers with data — skipping plot.")
            else:
                plot_heatmap(
                    matrix=sub_mat,
                    labels=sub_labels,
                    output_dir=lang_dir,
                    filename_stem=f"vocab_overlap_{lang}",
                    title=None if args.no_title else f"Vocabulary Overlap — {lang}",
                    dpi=args.dpi,
                )

        # Clean up caches after each language
        if not args.dry_run:
            for cd in (cache_fixed, cache_sslm, cache_hnet):
                shutil.rmtree(cd, ignore_errors=True)

    # ── aggregate matrix ──
    agg_matrix = np.where(count_matrix > 0, sum_matrix / count_matrix, 0.0)

    # Drop tokenizers that were never available for any language
    # (diagonal count_matrix[i,i] == 0 means tokenizer i never had data)
    agg_valid = [count_matrix[i, i] > 0 for i in range(n)]
    skipped_agg = [labels[i] for i, v in enumerate(agg_valid) if not v]
    if skipped_agg:
        print(f"\n[aggregate] Dropping from plot (no data in any language): {skipped_agg}")
    sub_agg, sub_agg_labels = filter_by_valid(agg_matrix, labels, agg_valid)

    print(f"\n{'='*60}")
    print("Aggregate pairwise Jaccard overlap matrix:")
    print(f"{'='*60}")
    if sub_agg_labels:
        col_w = max(len(l) for l in sub_agg_labels) + 2
        header = " " * col_w + "".join(f"{l:>{col_w}}" for l in sub_agg_labels)
        print(header)
        m = len(sub_agg_labels)
        for i, row_label in enumerate(sub_agg_labels):
            row_str = f"{row_label:<{col_w}}"
            for j in range(m):
                row_str += f"{sub_agg[i, j]:>{col_w}.3f}"
            print(row_str)
    else:
        print("  (no tokenizers with data)")

    # ── plot aggregate heatmap ──
    if sub_agg.size > 0:
        plot_heatmap(
            matrix=sub_agg,
            labels=sub_agg_labels,
            output_dir=args.output_dir,
            filename_stem="vocab_overlap",
            title=None if args.no_title else "Pairwise Vocabulary Overlap (Jaccard, averaged across languages)",
            dpi=args.dpi,
        )
    else:
        print("[aggregate] Nothing to plot — no tokenizer had data for any language.")

    # ── group-averaged plots ──────────────────────────────────────────────────
    def _plot_groups(group_map: dict, scheme_name: str):
        """Average lang_matrices within each group and save one plot per group."""
        # Discover which groups appear in the requested langs
        groups = {}
        for lang in args.langs:
            grp = group_map.get(lang)
            if grp is None:
                continue
            groups.setdefault(grp, []).append(lang)

        group_dir = os.path.join(args.output_dir, "by_group")
        print(f"\n{'='*60}")
        print(f"Group-averaged plots ({scheme_name})")
        print(f"{'='*60}")

        for grp, grp_langs in sorted(groups.items()):
            grp_sum   = np.zeros((n, n), dtype=float)
            grp_count = np.zeros((n, n), dtype=int)

            for lang in grp_langs:
                mat = lang_matrices.get(lang)
                if mat is None:
                    continue
                valid = ~np.isnan(mat)
                grp_sum[valid]   += mat[valid]
                grp_count[valid] += 1

            if grp_count.max() == 0:
                print(f"  [{grp}] No data — skipping.")
                continue

            grp_mat   = np.where(grp_count > 0, grp_sum / grp_count, 0.0)
            grp_valid = [grp_count[i, i] > 0 for i in range(n)]
            skipped   = [labels[i] for i, v in enumerate(grp_valid) if not v]
            if skipped:
                print(f"  [{grp}] Dropping (no data): {skipped}")
            sub_mat, sub_lbl = filter_by_valid(grp_mat, labels, grp_valid)

            if sub_mat.size == 0:
                print(f"  [{grp}] Nothing to plot — skipping.")
                continue

            # Safe filename: replace spaces / & with underscores
            safe_name = grp.lower().replace(" & ", "_").replace(" ", "_")
            stem = f"vocab_overlap_{scheme_name}_{safe_name}"
            title_str = f"Vocabulary Overlap — {grp} ({scheme_name.title()}, {len(grp_langs)} lang{'s' if len(grp_langs) != 1 else ''})"
            print(f"  Plotting group '{grp}' ({grp_langs}) …")
            plot_heatmap(
                matrix=sub_mat,
                labels=sub_lbl,
                output_dir=group_dir,
                filename_stem=stem,
                title=None if args.no_title else title_str,
                dpi=args.dpi,
            )

    if args.group_by in ("typology", "both"):
        _plot_groups(LANG_GROUPS, "typology")
    if args.group_by in ("script", "both"):
        _plot_groups(LANG_SCRIPTS, "script")

    # ── std-dev plots across languages within each group ─────────────────────
    if args.compute_stdev:
        def _plot_stdev_groups(group_map: dict, scheme_name: str):
            """For each group compute per-cell std-dev of Jaccard across languages.

            For each (tokenizer_i, tokenizer_j) cell we collect the Jaccard values
            from every language that belongs to the group, then take the sample
            std-dev (ddof=1 when >= 2 observations, else NaN).
            """
            groups = {}
            for lang in args.langs:
                grp = group_map.get(lang)
                if grp is None:
                    continue
                groups.setdefault(grp, []).append(lang)

            stdev_dir = os.path.join(args.output_dir, "stdev_by_group")
            print(f"\n{'='*60}")
            print(f"Std-dev plots ({scheme_name})")
            print(f"{'='*60}")

            for grp, grp_langs in sorted(groups.items()):
                # Stack per-language matrices: shape (num_langs, n, n)
                lang_mats = []
                for lang in grp_langs:
                    mat = lang_matrices.get(lang)
                    if mat is not None:
                        lang_mats.append(mat)

                if len(lang_mats) < 2:
                    print(f"  [{grp}] Only {len(lang_mats)} language(s) — need ≥ 2 for std-dev. Skipping.")
                    continue

                stacked = np.stack(lang_mats, axis=0)  # (L, n, n)
                # np.nanstd with ddof=1 for sample std-dev
                with np.errstate(invalid="ignore"):
                    stdev_mat = np.nanstd(stacked, axis=0, ddof=1)  # (n, n)

                # Where fewer than 2 non-NaN observations exist, set to NaN
                obs_count = np.sum(~np.isnan(stacked), axis=0)
                stdev_mat = np.where(obs_count >= 2, stdev_mat, np.nan)

                # Determine which tokenizers had data in >= 1 language
                grp_valid = [obs_count[i, i] >= 1 for i in range(n)]
                skipped = [labels[i] for i, v in enumerate(grp_valid) if not v]
                if skipped:
                    print(f"  [{grp}] Dropping (no data): {skipped}")

                idx = [i for i, v in enumerate(grp_valid) if v]
                if not idx:
                    print(f"  [{grp}] Nothing to plot — skipping.")
                    continue

                sub_stdev = stdev_mat[np.ix_(idx, idx)]
                sub_lbl   = [labels[i] for i in idx]

                safe_name = grp.lower().replace(" & ", "_").replace(" ", "_")
                stem  = f"vocab_overlap_stdev_{scheme_name}_{safe_name}"
                title_str = (
                    f"Jaccard Overlap Std-Dev — {grp} "
                    f"({scheme_name.title()}, {len(lang_mats)} lang{'s' if len(lang_mats) != 1 else ''})"
                )
                print(f"  Plotting std-dev for group '{grp}' ({grp_langs}) …")
                plot_stdev_heatmap(
                    stdev_matrix=sub_stdev,
                    labels=sub_lbl,
                    output_dir=stdev_dir,
                    filename_stem=stem,
                    title=None if args.no_title else title_str,
                    dpi=args.dpi,
                )

        if args.group_by in ("typology", "both"):
            _plot_stdev_groups(LANG_GROUPS, "typology")
        if args.group_by in ("script", "both"):
            _plot_stdev_groups(LANG_SCRIPTS, "script")

    # ── combined plots (overlap lower-left + stdev upper-right) ─────────────
    if args.combined:
        def _plot_combined_groups():
            """For each broad typology group, compute group-mean and group-stdev
            matrices then call plot_combined_heatmap.

            Two-pass approach: first collect all per-group data and find the
            global stdev maximum so both groups share the same colour scale.
            """
            groups = {}
            for lang in args.langs:
                grp = BROAD_LANG_GROUPS.get(lang)
                if grp is None:
                    continue
                groups.setdefault(grp, []).append(lang)

            combined_dir = os.path.join(args.output_dir, "combined_by_group")
            print(f"\n{'='*60}")
            print("Combined overlap+stdev plots (broad typology groups)")
            print(f"{'='*60}")

            # ── Pass 1: build all group data ──────────────────────────────────
            group_data = {}   # grp -> {mean, stdev, idx, lbl, lang_mats, grp_langs}
            for grp, grp_langs in sorted(groups.items()):
                lang_mats = [
                    lang_matrices[lang]
                    for lang in grp_langs
                    if lang_matrices.get(lang) is not None
                ]
                if not lang_mats:
                    print(f"  [{grp}] No data — skipping.")
                    continue

                # mean
                grp_sum   = np.zeros((n, n), dtype=float)
                grp_count = np.zeros((n, n), dtype=int)
                for mat in lang_mats:
                    valid = ~np.isnan(mat)
                    grp_sum[valid]   += mat[valid]
                    grp_count[valid] += 1
                grp_mean = np.where(grp_count > 0, grp_sum / grp_count, 0.0)

                # stdev
                if len(lang_mats) >= 2:
                    stacked = np.stack(lang_mats, axis=0)
                    with np.errstate(invalid="ignore"):
                        stdev_mat = np.nanstd(stacked, axis=0, ddof=1)
                    obs_count = np.sum(~np.isnan(stacked), axis=0)
                    stdev_mat = np.where(obs_count >= 2, stdev_mat, np.nan)
                else:
                    stdev_mat = np.full((n, n), np.nan)

                grp_valid = [grp_count[i, i] > 0 for i in range(n)]
                idx = [i for i, v in enumerate(grp_valid) if v]
                skipped = [labels[i] for i, v in enumerate(grp_valid) if not v]
                if skipped:
                    print(f"  [{grp}] Dropping (no data): {skipped}")
                if not idx:
                    print(f"  [{grp}] Nothing to plot — skipping.")
                    continue

                group_data[grp] = dict(
                    mean=grp_mean[np.ix_(idx, idx)],
                    stdev=stdev_mat[np.ix_(idx, idx)],
                    idx=idx,
                    lbl=[labels[i] for i in idx],
                    lang_mats=lang_mats,
                    grp_langs=grp_langs,
                )

            # ── Compute shared stdev vmax across ALL groups ────────────────────
            all_stdev_vals = np.concatenate([
                d["stdev"][~np.isnan(d["stdev"])].ravel()
                for d in group_data.values()
                if not np.all(np.isnan(d["stdev"]))
            ]) if group_data else np.array([])
            global_vmax_stdev = float(np.max(all_stdev_vals)) if all_stdev_vals.size > 0 else 1.0
            global_vmax_stdev = max(global_vmax_stdev, 1e-6)
            print(f"  Shared stdev colour scale: 0 – {global_vmax_stdev:.4f}")

            # ── Pass 2: plot with shared colour scale ─────────────────────────
            for grp, d in sorted(group_data.items()):
                safe_name = grp.lower().replace(" & ", "_").replace(" ", "_")
                stem = f"vocab_overlap_combined_{safe_name}"
                title_str = (
                    f"Vocabulary Overlap — {grp} "
                    f"({len(d['lang_mats'])} lang{'s' if len(d['lang_mats']) != 1 else ''})"
                )
                print(f"  Plotting combined for group '{grp}' ({d['grp_langs']}) …")
                plot_combined_heatmap(
                    mean_mat=d["mean"],
                    stdev_mat=d["stdev"],
                    labels=d["lbl"],
                    output_dir=combined_dir,
                    filename_stem=stem,
                    title=None if args.no_title else title_str,
                    dpi=args.dpi,
                    vmax_stdev=global_vmax_stdev,
                )

        _plot_combined_groups()

    print("\nDone.")


if __name__ == "__main__":
    main()

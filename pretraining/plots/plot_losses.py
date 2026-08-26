"""
plot_losses.py
--------------
Loads WandB-exported CSVs for train/loss and eval/loss vs epoch and produces
per-language publication-quality plots, in two variants:

  • Loss (cross-entropy)      → y = loss
  • Perplexity                → y = exp(loss)

Output files (saved in the same directory as this script):
  train_loss_{lang}.png          train_ppl_{lang}.png
  eval_loss_{lang}.png           eval_ppl_{lang}.png

Usage:
    python3 plot_losses.py
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Paths & configuration
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
TRAIN_CSV  = SCRIPT_DIR / "train_loss_vs_epoch.csv"
EVAL_CSV   = SCRIPT_DIR / "eval_loss_vs_epoch.csv"

# Consistent colour per tokeniser (add more as needed)
TOKENIZER_COLORS = {
    "bpe":          "#4361EE",
    "bpe-dropout":  "#3A0CA3",
    "boundlessbpe": "#7209B7",
    "pickybpe":     "#B5179E",
    "superbpe":     "#F72585",
    "morphbpe":     "#E85D04",
    "morphwp":      "#F48C06",
    "morphulm":     "#FAA307",
    "wordpiece":    "#2DC653",
    "unigram":      "#0077B6",
    "sage":         "#E63946",
    "pathpiece":    "#457B9D",
    "sslm-bpe":     "#BC4749",
    "sslm-wp":      "#F4A261",
    "sslm-ulm":     "#A8DADC",
}

# Marker cycle so overlapping lines are still distinguishable in B&W prints
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "p", "8", "<", ">", "H", "+"]

LANG_LABELS = {"eng": "English", "hin": "Hindi", "tel": "Telugu"}
LANG_ORDER  = ["eng", "hin", "tel"]

TRAIN_SMOOTHING = 30   # rolling window for the dense train CSV; set 1 to disable

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def parse_run_name(col: str):
    """Return (lang, tokenizer) from a column like 'eng_sage_10000 - train/loss',
    or None for MIN/MAX/_step columns."""
    m = re.match(r"^([a-z]+)_(.+?)_10000 - (?:train|eval)/loss$", col)
    return (m.group(1), m.group(2)) if m else None


def load_loss_csv(path: Path) -> dict:
    """
    Returns  { lang: { tokenizer: pd.Series(loss, index=epoch) } }
    """
    df = pd.read_csv(path)
    epoch_col = df.columns[0]
    data: dict = {}
    for col in df.columns[1:]:
        parsed = parse_run_name(col)
        if parsed is None:
            continue
        lang, tok = parsed
        series = (
            df[[epoch_col, col]]
            .rename(columns={epoch_col: "epoch", col: "loss"})
            .dropna()
            .astype(float)
            .set_index("epoch")["loss"]
            .sort_index()
        )
        if not series.empty:
            data.setdefault(lang, {})[tok] = series
    return data


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    return pd.Series(y).rolling(window, min_periods=1, center=True).mean().values


# ─────────────────────────────────────────────────────────────────────────────
# Core plotting function
# ─────────────────────────────────────────────────────────────────────────────

def plot_single_language(
    lang: str,
    lang_data: dict,           # { tokenizer: pd.Series(loss, index=epoch) }
    *,
    title: str,
    xlabel: str = "Epoch",
    ylabel: str,
    use_perplexity: bool = False,
    smoothing_window: int = 1,
    figsize: tuple = (5.5, 5.5),
) -> plt.Figure:
    """
    Draw one figure for a single language showing all tokenizer curves.
    If use_perplexity=True, y-values are exponentiated (ppl = exp(loss)).
    """
    fig, ax = plt.subplots(figsize=figsize)

    all_toks = sorted(lang_data.keys())
    marker_cycle = (MARKERS * 10)[:len(all_toks)]

    for tok, marker in zip(all_toks, marker_cycle):
        series = lang_data[tok]
        x = series.index.values
        y = series.values

        y = smooth(y, smoothing_window)
        if use_perplexity:
            y = np.exp(y)

        ax.plot(
            x, y,
            label=tok,
            color=TOKENIZER_COLORS.get(tok, "#888888"),
            linewidth=1.8,
            alpha=0.9,
            marker=marker,
            markevery=max(1, len(x) // 12),   # show ~12 markers per line
            markersize=5,
        )

    # ── Axes styling ──────────────────────────────────────────────────────────
    ax.set_title(title, fontsize=15, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if use_perplexity:
        ax.set_ylim(0, 25)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend = ax.legend(
        title="Tokenizer",
        title_fontsize=10,
        fontsize=9,
        ncol=max(1, len(all_toks) // 8 + 1),
        loc="upper right",
        framealpha=0.85,
        edgecolor="#cccccc",
    )

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading CSVs …")
    train_data = load_loss_csv(TRAIN_CSV)
    eval_data  = load_loss_csv(EVAL_CSV)

    # ── Define what to plot ──────────────────────────────────────────────────
    # Each entry: (data_dict, prefix, ylabel_loss, ylabel_ppl, smoothing, desc)
    plot_sets = [
        (
            train_data,
            "train",
            "Train Loss",
            "Train Perplexity",
            TRAIN_SMOOTHING,
            "Training",
        ),
        (
            eval_data,
            "eval",
            "Validation Loss",
            "Validation Perplexity",
            1,               # eval has only ~5 rows per run → no smoothing
            "Validation",
        ),
    ]

    saved: list[str] = []

    for data, prefix, ylabel_loss, ylabel_ppl, sw, desc_word in plot_sets:
        langs_present = [l for l in LANG_ORDER if l in data]

        for lang in langs_present:
            lang_label = LANG_LABELS.get(lang, lang)
            lang_data  = data[lang]

            # ── Loss plot ────────────────────────────────────────────────────
            fig_loss = plot_single_language(
                lang,
                lang_data,
                title=f"{desc_word} Loss vs. Epoch — {lang_label}",
                ylabel=ylabel_loss,
                use_perplexity=False,
                smoothing_window=sw,
            )
            for ext, kw in [("png", {"dpi": 200}), ("pdf", {})]:
                out = SCRIPT_DIR / f"{prefix}_loss_{lang}.{ext}"
                fig_loss.savefig(out, bbox_inches="tight", **kw)
                saved.append(str(out))
            plt.close(fig_loss)

            # ── Perplexity plot ──────────────────────────────────────────────
            fig_ppl = plot_single_language(
                lang,
                lang_data,
                title=f"{desc_word} Perplexity vs. Epoch — {lang_label}",
                ylabel=ylabel_ppl,
                use_perplexity=True,
                smoothing_window=sw,
            )
            for ext, kw in [("png", {"dpi": 200}), ("pdf", {})]:
                out = SCRIPT_DIR / f"{prefix}_ppl_{lang}.{ext}"
                fig_ppl.savefig(out, bbox_inches="tight", **kw)
                saved.append(str(out))
            plt.close(fig_ppl)

    print("\nSaved files:")
    for p in saved:
        print(f"  {p}")
    print("\nDone.")


if __name__ == "__main__":
    main()

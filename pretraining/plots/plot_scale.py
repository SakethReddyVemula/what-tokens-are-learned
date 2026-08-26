"""
plot_scale.py
-------------
Reads a manually-filled CSV (scale_results.csv) and produces publication-quality
scale-experiment plots for five metrics:

  1. Perplexity              (lower is better)
  2. Sentiment Analysis Accuracy
  3. POS Tagging F1-score
  4. NER F1-score
  5. Dependency Parsing LAS

Each plot has:
  • X-axis  : Metric value
  • Y-axis  : Model size (in millions of parameters, log scale)
  • Lines   : One per language, coloured by language
  • Style   : Solid  = BPE
              Dotted = SSLM-BPE

Output files (saved alongside this script):
  scale_perplexity.{png,pdf}
  scale_sentiment_accuracy.{png,pdf}
  scale_pos_f1.{png,pdf}
  scale_ner_f1.{png,pdf}
  scale_dep_las.{png,pdf}

Usage:
    # Copy scale_results_template.csv → scale_results.csv, fill in values, then:
    python3 plot_scale.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = Path(__file__).parent
RESULTS_CSV = SCRIPT_DIR / "scale_results.csv"

# ── Language colours (mirrors the tokenizer-colour philosophy from plot_losses) ──
LANG_COLORS = {
    "eng": "#4361EE",   # vivid blue
    "hin": "#F72585",   # magenta-pink
    "tel": "#2DC653",   # vivid green
}

LANG_LABELS = {
    "eng": "English",
    "hin": "Hindi",
    "tel": "Telugu",
}

LANG_ORDER = ["eng", "hin", "tel"]

# ── Tokenizer → line style ────────────────────────────────────────────────────
TOK_STYLE = {
    "bpe":      {"linestyle": "-",     "linewidth": 2.0},   # solid
    "sslm-bpe": {"linestyle": (0, (5, 3)), "linewidth": 2.0},   # custom dash (dotted-ish)
}
TOK_LABELS = {
    "bpe":      "BPE",
    "sslm-bpe": "SSLM-BPE",
}
TOK_ORDER = ["bpe", "sslm-bpe"]

# ── Marker per language (for B&W-printable distinction) ──────────────────────
LANG_MARKERS = {
    "eng": "o",
    "hin": "s",
    "tel": "^",
}

# ─────────────────────────────────────────────────────────────────────────────
# Metric definitions
# ─────────────────────────────────────────────────────────────────────────────

METRICS = [
    {
        "col":          "perplexity",
        "ylabel":       "Validation Perplexity",
        "title":        "Perplexity vs. Model Size",
        "out_stem":     "scale_perplexity",
        "invert_y":     False,
        "y_min":        0,      # start Y-axis at 0 (origin)
        "y_pct":        False,
    },
    {
        "col":          "sentiment_accuracy",
        "ylabel":       "Accuracy (%)",
        "title":        "Sentiment Analysis Accuracy vs. Model Size",
        "out_stem":     "scale_sentiment_accuracy",
        "invert_y":     False,
        "y_pct":        True,
    },
    {
        "col":          "pos_f1",
        "ylabel":       "F1-score (%)",
        "title":        "POS Tagging F1-score vs. Model Size",
        "out_stem":     "scale_pos_f1",
        "invert_y":     False,
        "y_pct":        True,
    },
    {
        "col":          "ner_f1",
        "ylabel":       "F1-score (%)",
        "title":        "NER F1-score vs. Model Size",
        "out_stem":     "scale_ner_f1",
        "invert_y":     False,
        "y_pct":        True,
    },
    {
        "col":          "dep_las",
        "ylabel":       "LAS",
        "title":        "Dependency Parsing LAS vs. Model Size",
        "out_stem":     "scale_dep_las",
        "invert_y":     False,
        "y_pct":        True,
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_results(path: Path) -> pd.DataFrame:
    """
    Load the results CSV, skipping comment lines that begin with '#'.
    Returns a cleaned DataFrame with columns:
        language, tokenizer, model_size_M, perplexity,
        sentiment_accuracy, pos_f1, ner_f1, dep_las
    """
    rows = []
    with open(path, encoding="utf-8") as fh:
        header = None
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if header is None:
                header = parts
                continue
            if len(parts) != len(header):
                continue
            rows.append(parts)

    if not rows:
        raise ValueError(
            f"No data rows found in {path}.\n"
            "Please fill in 'scale_results.csv' from the template."
        )

    df = pd.DataFrame(rows, columns=header)
    numeric_cols = ["model_size_M", "perplexity",
                    "sentiment_accuracy", "pos_f1", "ner_f1", "dep_las"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Core plotting function
# ─────────────────────────────────────────────────────────────────────────────

def plot_metric(
    df: pd.DataFrame,
    metric_cfg: dict,
    *,
    figsize: tuple = (6.0, 5.5),
) -> plt.Figure:
    """
    Produce one scale-experiment figure for the given metric.

    X-axis : model size (linear, ticks every 5 M, starts at 0)
    Y-axis : metric value
    Lines  : (language × tokenizer) combinations
             colour → language,  linestyle → tokenizer
    """
    import matplotlib.ticker as mticker

    col     = metric_cfg["col"]
    ylabel  = metric_cfg["ylabel"]
    title   = metric_cfg["title"]
    invert  = metric_cfg.get("invert_y", False)

    fig, ax = plt.subplots(figsize=figsize)

    # Keep only rows that have data for this metric
    sub = df.dropna(subset=[col, "model_size_M"]).copy()
    sub = sub.sort_values("model_size_M")

    for lang in LANG_ORDER:
        if lang not in df["language"].values:
            continue
        lang_df = sub[sub["language"] == lang]
        if lang_df.empty:
            continue

        color  = LANG_COLORS.get(lang, "#888888")
        marker = LANG_MARKERS.get(lang, "o")

        for tok in TOK_ORDER:
            tok_df = lang_df[lang_df["tokenizer"] == tok].dropna(subset=[col])
            if tok_df.empty:
                continue

            x = tok_df["model_size_M"].values
            y = tok_df[col].values

            style = TOK_STYLE.get(tok, {"linestyle": "-", "linewidth": 2.0})
            ax.plot(
                x, y,
                color=color,
                label=f"{LANG_LABELS.get(lang, lang)} — {TOK_LABELS.get(tok, tok)}",
                marker=marker,
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=0.8,
                linewidth=style["linewidth"],
                linestyle=style["linestyle"],
                alpha=0.92,
            )

    # ── Axes styling ──────────────────────────────────────────────────────────
    ax.set_title(title, fontsize=14, fontweight="bold", pad=11)
    ax.set_xlabel("Model Size (in million parameters)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    # Linear X-axis: step of 5, start from 0, labels like "0M", "5M", "10M" …
    max_size = sub["model_size_M"].max() if not sub.empty else 30
    ax.set_xlim(left=0, right=max_size * 1.08)   # small right padding
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{int(v)}M")
    )

    if invert:
        ax.invert_yaxis()

    y_min = metric_cfg.get("y_min", None)
    if y_min is not None:
        ax.set_ylim(bottom=y_min)

    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Compound legend: language colour swatches + tokenizer style swatches ──
    lang_handles = [
        mlines.Line2D([], [], color=LANG_COLORS.get(lg, "#888888"),
                      marker=LANG_MARKERS.get(lg, "o"), markersize=7,
                      markeredgecolor="white", markeredgewidth=0.8,
                      linewidth=2.0, linestyle="-",
                      label=LANG_LABELS.get(lg, lg))
        for lg in LANG_ORDER
        if lg in df["language"].values
    ]
    tok_handles = [
        mlines.Line2D([], [], color="#555555",
                      linewidth=2.0,
                      linestyle=TOK_STYLE.get(tk, {}).get("linestyle", "-"),
                      label=TOK_LABELS.get(tk, tk))
        for tk in TOK_ORDER
    ]

    # Two-section legend: languages first, then a separator, then tokenizer styles
    separator = mlines.Line2D([], [], color="none", label=" ")
    legend = ax.legend(
        handles=lang_handles + [separator] + tok_handles,
        fontsize=9,
        loc="best",
        framealpha=0.88,
        edgecolor="#cccccc",
        handlelength=2.5,
    )

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Check data file ───────────────────────────────────────────────────────
    if not RESULTS_CSV.exists():
        template = SCRIPT_DIR / "scale_results_template.csv"
        raise FileNotFoundError(
            f"Data file not found: {RESULTS_CSV}\n"
            f"Copy the template:\n"
            f"  cp '{template}' '{RESULTS_CSV}'\n"
            "then fill in your experimental results before running this script."
        )

    print(f"Loading data from {RESULTS_CSV} …")
    df = load_results(RESULTS_CSV)

    print(f"  Rows loaded : {len(df)}")
    print(f"  Languages   : {sorted(df['language'].unique())}")
    print(f"  Tokenizers  : {sorted(df['tokenizer'].unique())}")
    print(f"  Model sizes : {sorted(df['model_size_M'].dropna().unique())} M")
    print()

    saved: list[str] = []

    for metric_cfg in METRICS:
        col = metric_cfg["col"]
        if col not in df.columns or df[col].dropna().empty:
            print(f"  [skip] '{col}' — no data found, skipping this metric.")
            continue

        fig = plot_metric(df, metric_cfg)

        stem = metric_cfg["out_stem"]
        for ext, kw in [("png", {"dpi": 200}), ("pdf", {})]:
            out = SCRIPT_DIR / f"{stem}.{ext}"
            fig.savefig(out, bbox_inches="tight", **kw)
            saved.append(str(out))
        plt.close(fig)
        print(f"  Saved: {stem}.png / .pdf")

    if saved:
        print(f"\nAll done — {len(saved)} file(s) written.")
    else:
        print("\nNo plots were generated. Check your data file for valid numeric entries.")


if __name__ == "__main__":
    main()

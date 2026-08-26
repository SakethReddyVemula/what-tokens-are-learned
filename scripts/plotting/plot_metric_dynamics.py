#!/usr/bin/env python3
"""
plot_metric_dynamics.py
-----------------------
Plots how a tokenization metric evolves over SSLM training checkpoints
and draws a horizontal dashed line for H-Nets (best model only).

Supported metrics
-----------------
  compression         compression_rate in compression/<lang>/*.json
  exponence           overall_average_exponence in exponence/<lang>/*.json
  fertility           fertility in fertility/<lang>/*.json
  mean_token_length   computed from length_dist/<lang>/*.json (weighted mean)
  renyi_entropy       renyi_entropy in renyi_entropy/<lang>/*.json
  ttr                 ttr in ttr/<lang>/*.json
  vocab_size          effective_vocab_size in vocab_size/<lang>/*.json

SSLM files are named:  sslm_checkpoint_{epoch}_{step}_test.json
H-Net file is named:   hnet_best_model_test.json

Usage
-----
python plot_metric_dynamics.py \\
    --metric fertility \\
    --languages eng hin tam fin hun \\
    --input_dir evaluation_results \\
    --output_dir plots_dynamics \\
    --pct_steps          # x-axis as % of training steps (default: raw steps)
"""

import argparse
import json
import os
import re
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# # ── language metadata ──────────────────────────────────────────────────────────
# LANG_INFO = {
#     "fin": "Finnish (Agglutinative)",
#     "hun": "Hungarian (Agglutinative)",
#     "mal": "Malayalam (Agglutinative)",
#     "tam": "Tamil (Agglutinative)",
#     "tel": "Telugu (Agglutinative)",
#     "kir": "Kyrgyz (Agglutinative)",
#     "tur": "Turkish (Agglutinative)",
#     "mon": "Mongolian (Agglutinative)",
#     "ind": "Indonesian (Agglutinative)",
#     "san": "Sanskrit (Fusional)",
#     "hin": "Hindi (Fusional)",
#     "snd": "Sindhi (Fusional)",
#     "hrv": "Croatian (Fusional)",
#     "rus": "Russian (Fusional)",
#     "fas": "Persian (Fusional)",
#     "eng": "English (Analytic)",
#     "swe": "Swedish (Analytic)",
#     "heb": "Hebrew (Templatic)",
# }

# ── language metadata (by script) ──────────────────────────────────────────────────────────
LANG_INFO = {
    "fin": "Finnish (Latin)",
    "hun": "Hungarian (Latin)",
    "mal": "Malayalam (Dravidian)",
    "tam": "Tamil (Dravidian)",
    "tel": "Telugu (Dravidian)",
    "kir": "Kyrgyz (Cyrillic)",
    "tur": "Turkish (Latin)",
    "mon": "Mongolian (Cyrillic)",
    "ind": "Indonesian (Latin)",
    "san": "Sanskrit (Devanagiri)",
    "hin": "Hindi (Devanagiri)",
    "snd": "Sindhi (Arabic)",
    "hrv": "Croatian (Latin)",
    "rus": "Russian (Cyrillic)",
    "fas": "Persian (Arabic)",
    "eng": "English (Latin)",
    "swe": "Swedish (Latin)",
    "heb": "Hebrew (Hebrew)",
}

# Fixed colour / marker for each language (same palette as plot_f1.py style)
LANG_STYLE = {
    "eng": ("#1f77b4", "o"),
    "hin": ("#9467bd", "v"),
    "tam": ("#ff7f0e", "s"),
    "fin": ("#2ca02c", "4"),
    "hun": ("#d62728", "D"),
    "ind": ("#8c564b", "P"),
    "mal": ("#e377c2", "X"),
    "tel": ("#7f7f7f", "h"),
    "kir": ("#bcbd22", "<"),
    "tur": ("#17becf", ">"),
    "mon": ("#aec7e8", "H"),
    "san": ("#ffbb78", "8"),
    "snd": ("#98df8a", "p"),
    "hrv": ("#ff9896", "d"),
    "rus": ("#c5b0d5", "1"),
    "fas": ("#c49c94", "2"),
    "swe": ("#f7b6d2", "3"),
    "heb": ("#2ca02c", "^"),
}

# Fallback for languages not in the dict
_FALLBACK_COLORS = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
    "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
]
_FALLBACK_MARKERS = ["o","s","D","^","v","P","X","h","<",">"]

# ── metric configuration ───────────────────────────────────────────────────────
# Maps metric name → (results_subdirectory, key_extractor_function)
def _scalar_key(key):
    return lambda d: d[key]

def _mean_token_length(d):
    """Weighted mean of {length: count} distribution."""
    total_count = 0
    total_len = 0
    for k, v in d.items():
        length = int(k)
        count = int(v)
        total_count += count
        total_len += length * count
    if total_count == 0:
        return None
    return total_len / total_count

METRIC_CONFIG = {
    "compression":        ("compression",   _scalar_key("compression_rate")),
    "exponence":          ("exponence",     _scalar_key("overall_average_exponence")),
    "fertility":          ("fertility",     _scalar_key("fertility")),
    "mean_token_length":  ("length_dist",   _mean_token_length),
    "renyi_efficiency":      ("renyi_efficiency", _scalar_key("renyi_efficiency")),
    "renyi_entropy":      ("renyi_entropy", _scalar_key("renyi_entropy")),
    "ttr":                ("ttr",           _scalar_key("ttr")),
    "vocab_size":         ("vocab_size",    _scalar_key("effective_vocab_size")),
}

METRIC_LABELS = {
    "compression":       "Compression Rate",
    "exponence":         "Exponence",
    "fertility":         "Fertility",
    "mean_token_length": "Mean Token Length (chars)",
    "renyi_efficiency":  "Rényi Efficiency",
    "renyi_entropy":     "Rényi Entropy",
    "ttr":               "Type-Token Ratio (TTR)",
    "vocab_size":        "Effective Vocabulary Size",
}

# ── file name parsing ─────────────────────────────────────────────────────────

_SSLM_RE = re.compile(r"sslm_checkpoint_(\d+)_(\d+)_test\.json$")
_HNET_RE = re.compile(r"hnet_best_model_test\.json$")


def parse_step(filename):
    """Return global training step from an SSLM checkpoint filename, or None."""
    m = _SSLM_RE.search(filename)
    if m:
        return int(m.group(2))
    return None


def is_hnet(filename):
    return bool(_HNET_RE.search(filename))


# ── data loading ──────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_sslm_series(lang_dir, extractor):
    """
    Scan lang_dir for SSLM checkpoint JSON files.
    Returns sorted list of (step, value) pairs.
    """
    data = []
    if not os.path.isdir(lang_dir):
        return data
    for fname in os.listdir(lang_dir):
        step = parse_step(fname)
        if step is None:
            continue
        fpath = os.path.join(lang_dir, fname)
        try:
            d = load_json(fpath)
            val = extractor(d)
            if val is not None:
                data.append((step, val))
        except (KeyError, ValueError, TypeError) as e:
            print(f"  [warn] {fpath}: {e}", file=sys.stderr)
    data.sort(key=lambda x: x[0])
    return data


def load_hnet_value(lang_dir, extractor):
    """Return the H-Net best-model scalar value, or None if not found."""
    if not os.path.isdir(lang_dir):
        return None
    fpath = os.path.join(lang_dir, "hnet_best_model_test.json")
    if not os.path.isfile(fpath):
        return None
    try:
        d = load_json(fpath)
        return extractor(d)
    except (KeyError, ValueError, TypeError) as e:
        print(f"  [warn] {fpath}: {e}", file=sys.stderr)
        return None


# ── plotting ──────────────────────────────────────────────────────────────────

# Training phase bands (% of pretraining steps)
PHASES = [
    (0,   2.5,   "#d0e8fb", "Rapid evolution"),                          # light blue
    (2.5,   10,  "#fef3cd", "Fluctuation in\nagglutinative\nlanguages"),  # light amber
    (10,  25,  "#d4edda", "Convergence"),                               # light green
    (25,  100, "#f8d7da", "Saturation"),                                # light red/pink
]

def lang_style(lang_code, fallback_idx):
    if lang_code in LANG_STYLE:
        return LANG_STYLE[lang_code]
    color  = _FALLBACK_COLORS[fallback_idx % len(_FALLBACK_COLORS)]
    marker = _FALLBACK_MARKERS[fallback_idx % len(_FALLBACK_MARKERS)]
    return color, marker


def plot_metric_dynamics(
    metric: str,
    languages: list,
    input_dir: str,
    output_dir: str,
    figsize: tuple,
    title: str | None,
    dpi: int,
    pct_steps: bool,
    no_legend: bool,
):
    if metric not in METRIC_CONFIG:
        sys.exit(
            f"Unknown metric '{metric}'. Valid options: "
            + ", ".join(sorted(METRIC_CONFIG))
        )

    subdir, extractor = METRIC_CONFIG[metric]
    metric_dir = os.path.join(input_dir, subdir)
    if not os.path.isdir(metric_dir):
        sys.exit(f"Metric directory not found: {metric_dir}")

    # Discover languages
    available_langs = sorted(os.listdir(metric_dir))
    if languages:
        selected = [l for l in languages if l in available_langs]
        missing  = [l for l in languages if l not in available_langs]
        if missing:
            print(f"  [warn] Languages not found in {metric_dir}: {missing}",
                  file=sys.stderr)
    else:
        selected = available_langs

    if not selected:
        sys.exit("No languages to plot after filtering.")

    fig_w, fig_h = figsize
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    plotted_sslm = 0
    plotted_hnet = 0
    fallback_idx = 0

    for lang in selected:
        lang_dir = os.path.join(metric_dir, lang)
        color, marker = lang_style(lang, fallback_idx)
        fallback_idx += 1
        label = LANG_INFO.get(lang, lang)

        # ── SSLM dynamics ──
        sslm_data = load_sslm_series(lang_dir, extractor)
        if sslm_data:
            steps, vals = zip(*sslm_data)
            steps = np.array(steps, dtype=float)
            vals  = np.array(vals,  dtype=float)

            x = (steps / steps.max() * 100) if pct_steps else steps

            markevery = max(1, len(x) // 10)
            ax.plot(
                x, vals,
                color=color,
                linestyle="-",
                marker=marker,
                markersize=3.5,
                linewidth=1.5,
                alpha=0.90,
                markevery=markevery,
                # label=f"{label} (SSLMs)",
                label=f"{label}",
                zorder=3,
            )
            plotted_sslm += 1

        # ── H-Net horizontal line ──
        # hnet_val = load_hnet_value(lang_dir, extractor)
        # if hnet_val is not None:
        #     # Draw over the full x-range; we'll adjust xlim afterwards
        #     ax.axhline(
        #         hnet_val,
        #         color=color,
        #         linestyle="--",
        #         linewidth=1.2,
        #         alpha=0.75,
        #         label=f"{label} (H-Nets)",
        #         zorder=2,
        #     )
        #     plotted_hnet += 1

    if plotted_sslm == 0 and plotted_hnet == 0:
        sys.exit("No data found. Check --input_dir and --languages.")

    # ── axes ──
    xlabel = "% Pretraining Steps" if pct_steps else "Training Step"
    ax.set_xlabel(xlabel, fontsize=12.5)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric.replace("_", " ").title()),
                  fontsize=12.5)

    # ── Background phase bands (only for % steps) ──
    if pct_steps:
        for x0, x1, fc, label in PHASES:
            ax.axvspan(x0, x1, color=fc, alpha=0.45, zorder=0, linewidth=0)
            ax.text(
                x0 + (x1 - x0) * 0.08, 0.02, label,
                transform=ax.get_xaxis_transform(),
                ha='left', va='bottom',
                fontsize=6.5, color='#555555',
                style='italic', rotation=90,
                zorder=2,
            )

    if pct_steps:
        ax.set_xlim(0, 100)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%g%%'))
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.45, zorder=1)
    ax.set_axisbelow(True)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)

    # ── legend ──
    if not no_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            # Two conceptual     groups: SSLM lines, H-Net dashes
            ax.legend(
                handles=handles,
                labels=labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.01),
                ncol=3,
                fontsize=12.5,
                frameon=True,
                framealpha=0.95,
                edgecolor="#cccccc",
                columnspacing=1.0,
                handlelength=1.8,
                handletextpad=0.4,
                borderpad=0.5,
            )

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)

    base_name = f"{metric}_dynamics"
    pdf_path = os.path.join(output_dir, f"{base_name}.pdf")
    png_path = os.path.join(output_dir, f"{base_name}.png")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=dpi)
    plt.savefig(png_path, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"Saved → {pdf_path}  (and .png)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Plot tokenization metric dynamics over SSLM training checkpoints.
            H-Net best-model scores are shown as horizontal dashed lines.

            Supported metrics
            -----------------
              compression, exponence, fertility, mean_token_length,
              renyi_entropy, ttr, vocab_size

            Examples
            --------
            # Fertility dynamics for all available languages
            python plot_metric_dynamics.py --metric fertility

            # Subset of languages, x-axis as % of training
            python plot_metric_dynamics.py --metric ttr \\
                --languages eng hin tam fin hun \\
                --pct_steps
        """),
    )
    parser.add_argument(
        "--metric", required=True,
        help=(
            "Metric to plot. One of: "
            + ", ".join(sorted(METRIC_CONFIG))
        ),
    )
    parser.add_argument(
        "--languages", nargs="*", default=None,
        help="Language codes to include (default: all found in metric dir).",
    )
    parser.add_argument(
        "--input_dir", default="evaluation_results",
        help="Root directory of evaluation results (default: evaluation_results).",
    )
    parser.add_argument(
        "--output_dir", default="plots_dynamics",
        help="Directory to write plots to (default: plots_dynamics).",
    )
    parser.add_argument(
        "--figsize", nargs=2, type=float, default=[6.0, 4.0],
        metavar=("W", "H"),
        help="Figure size in inches (default: 6 4).",
    )
    parser.add_argument(
        "--title", default=None,
        help="Custom plot title.",
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help="Resolution for saved figures (default: 200).",
    )
    parser.add_argument(
        "--pct_steps", action="store_true",
        help="Show x-axis as %% of total training steps instead of raw steps.",
    )
    parser.add_argument(
        "--no_legend", action="store_true",
        help="Omit the legend.",
    )
    args = parser.parse_args()

    plot_metric_dynamics(
        metric=args.metric,
        languages=args.languages,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        figsize=tuple(args.figsize),
        title=args.title,
        dpi=args.dpi,
        pct_steps=args.pct_steps,
        no_legend=args.no_legend,
    )


if __name__ == "__main__":
    main()

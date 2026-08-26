#!/usr/bin/env python3
"""
plot_graphs.py
--------------
Reads evaluation CSVs produced by create_csv.py and draws a horizontal
dot-plot (Cleveland-style):

  • X-axis  : metric value
  • Y-axis  : language labels
  • Series  : tokenizer / config combinations, each with a unique marker + color
  • Special : SSLM and H-Nets points are drawn with a star ("*") marker and
              are explicitly highlighted via larger size and a black edge.

Usage
-----
python plot_graphs.py \\
    --metric fertility \\
    --languages fin hun tam tel \\
    --tokenizers bpe unigram sslm hnet \\
    --input_dir evaluation_csv_results \\
    --output_dir plots \\
    --vocab_sizes 10000          # optional – keep only columns with this config
"""

import argparse
import csv
import os
import re
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from matplotlib.transforms import blended_transform_factory

# ── language metadata ──────────────────────────────────────────────────────────
LANG_INFO = {
    "fin": "Finnish",
    "hun": "Hungarian",
    "mal": "Malayalam",
    "tam": "Tamil",
    "tel": "Telugu",
    "kir": "Kyrgyz",
    "tur": "Turkish",
    "mon": "Mongolian",
    "ind": "Indonesian",
    "san": "Sanskrit",
    "hin": "Hindi",
    "snd": "Sindhi",
    "hrv": "Croatian",
    "rus": "Russian",
    "fas": "Persian",
    "eng": "English",
    "swe": "Swedish",
    "heb": "Hebrew",
}

STANDARD_LANG_ORDER = [
    "fin", "hun", "mal", "tam", "tel", "kir", "tur", "mon",
    "ind", "san", "hin", "snd", "hrv", "rus", "fas", "eng", "swe", "heb",
]

# ── tokenizer display names ────────────────────────────────────────────────────
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

# Tokenizers that should receive a star marker
STAR_TOKENIZERS = {"SSLM", "H-Nets"}

# ── aesthetics ─────────────────────────────────────────────────────────────────
# A large, visually distinct palette
PALETTE = [
    "#b81c1e", "#1f5fa6", "#2d8a2d", "#6a2d8a", "#c85a00",
    "#7a3a18", "#c0408a", "#555555", "#2a9d7a", "#d95e30",
    "#5b6faa", "#c05090", "#6aaa28", "#ccaa00", "#777777",
    "#0d7a55", "#a83e00", "#4a4890", "#b01060", "#3d7a10",
]

MARKERS_REGULAR = ["o", "s", "D", "^", "v", "P", "X", "h", "8", "p",
                   "<", ">", "H", "d", "1", "2", "3", "4"]

MARKER_STAR = "*"

# ── language typology groups ───────────────────────────────────────────────────
LANG_GROUPS = {
    "fin": "Agglutinative", "hun": "Agglutinative", "mal": "Agglutinative",
    "tam": "Agglutinative", "tel": "Agglutinative", "kir": "Agglutinative",
    "tur": "Agglutinative", "mon": "Agglutinative", "ind": "Agglutinative",
    "san": "Fusional", "hin": "Fusional", "snd": "Fusional",
    "hrv": "Fusional", "rus": "Fusional", "fas": "Fusional",
    "eng": "Analytic & Introflexive", "swe": "Analytic & Introflexive",
    "heb": "Analytic & Introflexive",
}

# Two alternating shades (even row, odd row) per group — kept light for readability
GROUP_PALETTE = {
    "Agglutinative":           ("#e8f3fb", "#d4e8f5"),   # light blue
    "Fusional":                ("#fef3e6", "#fce3c8"),   # light orange
    "Analytic & Introflexive": ("#edf7e8", "#d8eecc"),   # light green
}
GROUP_ORDER = ["Agglutinative", "Fusional", "Analytic & Introflexive"]

# ── script groups ─────────────────────────────────────────────────────
LANG_SCRIPTS = {
    "fin": "Latin",      "hun": "Latin",      "tur": "Latin",
    "ind": "Latin",      "hrv": "Latin",      "eng": "Latin",      "swe": "Latin",
    "kir": "Cyrillic",  "mon": "Cyrillic",  "rus": "Cyrillic",
    "san": "Devanagari","hin": "Devanagari",
    "snd": "Arabic",    "fas": "Arabic",
    "mal": "Malayalam",
    "tam": "Tamil",
    "tel": "Telugu",
    "heb": "Hebrew",
}

SCRIPT_PALETTE = {
    "Latin":      ("#e6eef8", "#ccddf0"),   # steel blue
    "Cyrillic":   ("#fce5cd", "#f7d0ae"),   # orange
    "Devanagari": ("#edf7e8", "#d8eecc"),   # green
    "Arabic":     ("#fde8f0", "#f5cede"),   # pink
    "Malayalam":  ("#f3ecfa", "#e2d1f4"),   # lavender
    "Tamil":      ("#fdfbe6", "#f5f0be"),   # yellow
    "Telugu":     ("#e6f8f5", "#c4eee8"),   # teal
    "Hebrew":     ("#fdf5e6", "#f5e4c0"),   # cream
}

# Language ordering when grouped by script (same-script languages adjacent)
SCRIPT_LANG_ORDER = [
    # Latin
    "fin", "hun", "tur", "ind", "hrv", "eng", "swe",
    # Cyrillic
    "kir", "mon", "rus",
    # Devanagari
    "san", "hin",
    # Arabic
    "snd", "fas",
    # Malayalam
    "mal",
    # Tamil
    "tam",
    # Telugu
    "tel",
    # Hebrew
    "heb",
]


# ── helpers ────────────────────────────────────────────────────────────────────

def normalise_tokenizer_key(raw: str) -> str:
    """Lower-case and strip to match TOKENIZER_MAP keys."""
    return raw.strip().lower()


def display_name(tok_key: str) -> str:
    return TOKENIZER_MAP.get(tok_key, tok_key.title())


def is_star(disp_name: str) -> bool:
    return disp_name in STAR_TOKENIZERS


def parse_headers(rows):
    """
    Given the first two CSV rows, return a list of dicts:
      [{"tokenizer": "BPE", "config": "v10000", "col_idx": 3}, ...]
    Columns 0-2 are Language / lang code / script – skip them.
    """
    row1, row2 = rows[0], rows[1]
    series = []
    current_tok = ""
    for i in range(3, len(row1)):
        tok = row1[i].strip()
        cfg = row2[i].strip() if i < len(row2) else ""
        if tok:
            current_tok = tok
        series.append({"tokenizer": current_tok, "config": cfg, "col_idx": i})
    return series


def load_csv(csv_path: str):
    """Return (header_rows, data_rows)."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if len(rows) < 2:
        sys.exit(f"CSV file too short: {csv_path}")
    return rows[:2], rows[2:]


def filter_series(series, tokenizer_filter, vocab_size_filter):
    """
    Keep only series entries matching the specified tokenizers and vocab sizes.
    tokenizer_filter : list of display names (case-insensitive) or None → keep all
    vocab_size_filter: list of strings like "10000" or None → keep all
    """
    kept = []
    for s in series:
        tok_key = normalise_tokenizer_key(s["tokenizer"])
        disp = display_name(tok_key)

        if tokenizer_filter:
            match = any(
                t.lower() in (tok_key, disp.lower(), disp.replace("-", "").lower())
                for t in tokenizer_filter
            )
            if not match:
                continue

        if vocab_size_filter:
            cfg = s["config"].replace("v", "")
            if cfg not in vocab_size_filter and s["config"] not in vocab_size_filter:
                # SSLM uses "highest_step" – always keep it regardless of vocab filter
                if s["config"] not in ("highest_step", "best_model"):
                    continue

        kept.append(s)
    return kept


def filter_languages(data_rows, lang_filter, lang_order=None):
    """
    Keep rows whose language code (col 1) matches lang_filter.
    Returns rows ordered by lang_order (default: STANDARD_LANG_ORDER).
    """
    if lang_order is None:
        lang_order = STANDARD_LANG_ORDER
    lang_set = {l.lower() for l in lang_filter} if lang_filter else None

    lang_rows = {}
    for row in data_rows:
        if len(row) < 2:
            continue
        code = row[1].strip().lower()
        if lang_set is None or code in lang_set:
            lang_rows[code] = row

    # Order by lang_order, then any extras alphabetically
    order = [c for c in lang_order if c in lang_rows]
    extras = sorted(c for c in lang_rows if c not in lang_order)
    return [lang_rows[c] for c in order + extras]


def build_series_label(tok_disp: str, config: str) -> str:
    if not config or config in ("highest_step", "best_model"):
        return tok_disp
    # config is like "v10000" or "10000"
    cfg = config.lstrip("v")
    if cfg.isdigit():
        # return f"{tok_disp} v{int(cfg):,}"
        return f"{tok_disp}"
    return f"{tok_disp} ({config})"


# ── main plotting ──────────────────────────────────────────────────────────────

def plot(
    metric: str,
    languages: list,          # list of lang codes
    tokenizers: list,         # list of raw tokenizer keys / display names
    vocab_sizes: list,        # list like ["10000"] or None
    input_dir: str,
    output_dir: str,
    figsize: tuple,
    title: str | None,
    dpi: int,
    no_legend: bool,
    group_by: str = "typology",  # "typology" or "script"
):
    # ── locate CSV ──
    csv_path = os.path.join(input_dir, metric, f"{metric}_results.csv")
    if not os.path.exists(csv_path):
        sys.exit(
            f"CSV not found: {csv_path}\n"
            f"Run create_csv.py --metric {metric} first."
        )

    header_rows, data_rows = load_csv(csv_path)
    all_series = parse_headers(header_rows)
    series = filter_series(all_series, tokenizers, vocab_sizes)

    if not series:
        sys.exit("No columns matched the given --tokenizers / --vocab_sizes filters.")

    # Choose language ordering and group definitions based on group_by
    if group_by == "script":
        lang_order     = SCRIPT_LANG_ORDER
        lang_group_map = LANG_SCRIPTS
        group_pal_map  = SCRIPT_PALETTE
    else:
        lang_order     = STANDARD_LANG_ORDER
        lang_group_map = LANG_GROUPS
        group_pal_map  = GROUP_PALETTE

    lang_rows = filter_languages(data_rows, languages, lang_order=lang_order)
    lang_rows = list(reversed(lang_rows))   # first language at top of Y-axis
    if not lang_rows:
        sys.exit("No language rows matched the given --languages filter.")

    # Y-axis labels (bottom → top, so we reverse for display)
    y_labels = []
    for row in lang_rows:
        code = row[1].strip().lower()
        y_labels.append(LANG_INFO.get(code, row[0].strip()))

    n_langs = len(y_labels)
    y_pos = np.arange(n_langs)

    # ── assign colours / markers ──
    # Group series by tokenizer display name → one colour per tokenizer
    unique_toks = list(dict.fromkeys(
        display_name(normalise_tokenizer_key(s["tokenizer"])) for s in series
    ))
    tok_color = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(unique_toks)}

    # Assign regular markers round-robin (skip star, used for special toks)
    reg_marker_pool = [m for m in MARKERS_REGULAR]
    tok_marker = {}
    reg_idx = 0
    for t in unique_toks:
        if is_star(t):
            tok_marker[t] = MARKER_STAR
        else:
            tok_marker[t] = reg_marker_pool[reg_idx % len(reg_marker_pool)]
            reg_idx += 1

    # ── figure geometry ──
    # 0.35 in per row keeps the figure compact
    fig_h = max(3, n_langs * 0.35 + 2.0)
    fig_w = figsize[0] if figsize else 12
    fig_h = figsize[1] if figsize else fig_h
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # ── group-colored background bands ──
    group_extents = {}   # group → [min_yi, max_yi]
    group_row_count = {} # track alternation within each group
    for yi, row in enumerate(lang_rows):
        code = row[1].strip().lower()
        grp = lang_group_map.get(code, "Unknown")
        colors = group_pal_map.get(grp, ("#f0f0f0", "#e0e0e0"))
        within = group_row_count.get(grp, 0)
        ax.axhspan(yi - 0.5, yi + 0.5, color=colors[within % 2], zorder=0, linewidth=0)
        group_row_count[grp] = within + 1
        if grp not in group_extents:
            group_extents[grp] = [yi, yi]
        else:
            group_extents[grp][0] = min(group_extents[grp][0], yi)
            group_extents[grp][1] = max(group_extents[grp][1], yi)

    # ── plot each series ──
    legend_handles = []
    plotted_labels = set()

    for s in series:
        tok_key = normalise_tokenizer_key(s["tokenizer"])
        disp = display_name(tok_key)
        color = tok_color[disp]
        marker = tok_marker[disp]
        label = build_series_label(disp, s["config"])
        star = is_star(disp)

        xs = []
        ys = []
        for yi, row in enumerate(lang_rows):
            ci = s["col_idx"]
            if ci >= len(row):
                continue
            val = row[ci].strip()
            if not val:
                continue
            try:
                xs.append(float(val))
                ys.append(yi)
            except ValueError:
                pass

        if not xs:
            continue

        ms = 12 if star else 7
        zord = 5 if star else 3
        lw = 1.2 if star else 0.6

        ax.scatter(
            xs, ys,
            marker=marker,
            color=color,
            s=ms ** 2,
            alpha=0.8,
            zorder=zord,
            linewidths=lw,
            edgecolors="black" if star else color,
            label=label if label not in plotted_labels else "_nolegend_",
        )

        if label not in plotted_labels:
            h = mlines.Line2D(
                [], [],
                color=color,
                marker=marker,
                linestyle="None",
                markersize=ms,
                markeredgecolor="black" if star else color,
                markeredgewidth=lw,
                label=label,
            )
            legend_handles.append(h)
            plotted_labels.add(label)

    # ── axes decoration ──
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_xlabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel("Language", fontsize=12)
    # ax.set_title(
    #     title or f"{metric.replace('_', ' ').title()} by Language and Tokenizer",
    #     fontsize=13, fontweight="bold", pad=12,
    # )
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    # Extend x range slightly
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin - (xmax - xmin) * 0.02, xmax + (xmax - xmin) * 0.04)
    ax.set_ylim(-0.8, n_langs - 0.2)

    # ── italic group labels at top-right of each group block ──
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for grp, (y_lo, y_hi) in group_extents.items():
        ax.text(
            0.995, y_hi + 0.45,
            grp,
            transform=trans,
            ha="right", va="top",
            fontsize=7.5, fontstyle="italic",
            color="#444444",
            zorder=7,
        )

    # ── legend (top, 8 columns) ──
    if not no_legend and legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01),
            borderaxespad=0,
            fontsize=10,
            ncol=8,
            framealpha=0.9,
            # title="Tokenizer",
            title_fontsize=8.5,
            handlelength=1.2,
            columnspacing=0.8,
        )

    plt.tight_layout()

    # ── save ──
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{metric}.pdf"
    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    # Also save PNG for quick inspection
    plt.savefig(os.path.join(output_dir, f"{metric}.png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}  (and .png)")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Plot evaluation metric results as horizontal dot-plots.

            Examples
            --------
            # All languages and tokenizers, fertility metric
            python plot_graphs.py --metric fertility

            # Subset of languages
            python plot_graphs.py --metric compression --languages fin hun tam tel

            # Subset of tokenizers
            python plot_graphs.py --metric morphscore_f1 \\
                --tokenizers bpe unigram sslm hnet morfessor

            # Only v10000 vocab-size columns
            python plot_graphs.py --metric fertility --vocab_sizes 10000
        """),
    )
    parser.add_argument("--metric", required=True,
                        help="Metric name, e.g. fertility, compression, morphscore_f1")
    parser.add_argument("--languages", nargs="*", default=None,
                        help="Language codes to include (default: all)")
    parser.add_argument("--tokenizers", nargs="*", default=None,
                        help="Tokenizer keys/display names to include (default: all)")
    parser.add_argument("--vocab_sizes", nargs="*", default=None,
                        help="Vocab sizes to keep, e.g. 10000 20000 (default: all)")
    parser.add_argument("--input_dir", default="evaluation_csv_results",
                        help="Directory containing metric sub-folders with CSVs")
    parser.add_argument("--output_dir", default="plots",
                        help="Directory to save plots")
    parser.add_argument("--figsize", nargs=2, type=float, default=None,
                        metavar=("W", "H"), help="Figure size in inches")
    parser.add_argument("--title", default=None,
                        help="Custom plot title (optional)")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Resolution for saved figures (default 150)")
    parser.add_argument("--no_legend", action="store_true",
                        help="Omit the legend")
    parser.add_argument("--group_by", default="typology",
                        choices=["typology", "script"],
                        help="Background grouping: 'typology' (default) or 'script'")
    args = parser.parse_args()

    plot(
        metric=args.metric,
        languages=args.languages,
        tokenizers=args.tokenizers,
        vocab_sizes=args.vocab_sizes,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        figsize=tuple(args.figsize) if args.figsize else None,
        title=args.title,
        dpi=args.dpi,
        no_legend=args.no_legend,
        group_by=args.group_by,
    )


if __name__ == "__main__":
    main()

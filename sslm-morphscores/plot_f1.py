import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# 5 representative languages with fixed colour / marker / label
#   Colour legend: Blue=Analytic, Green=Agglutinative, Red=Fusional,
#                  Gold=Templatic, Purple=Morphologically-rich
# ---------------------------------------------------------------------------
LANGUAGES = [
    # (csv_prefix,  display_label,                  color,     marker)
    ("eng_latn", "English (Analytic)",           "#1f77b4", "o"),
    ("tam_taml", "Tamil (Agglutinative)",       "#ff7f0e", "s"),
    ("heb_hebr", "Hebrew (Templatic)",            "#2ca02c", "^"),
    ("hun_latn", "Hungarian (Agglutinative)","#d62728", "D"),
    ("hin_deva", "Hindi (Fusional)",              "#9467bd", "v"),
    ("ind_latn", "Indonesian (Agglutinative)", "#8c564b", 'P')
]


def parse_checkpoint(name):
    """Return the global step integer from a checkpoint name string."""
    # 'lang/checkpoint_{epoch}_{step}.pt'
    m = re.search(r'checkpoint_(\d+)_(\d+)\.pt', str(name))
    if m:
        return int(m.group(2))
    # 'lang/checkpoint{epoch}.pt'  (end-of-epoch saves, step = epoch × steps_per_epoch)
    # We skip these to avoid duplicates on the x-axis.
    return None


def load_lang(csv_path):
    """Load CSV and return a clean DataFrame with columns [step, f1]."""
    df = pd.read_csv(csv_path)
    df['step'] = df['checkpoint_name'].apply(parse_checkpoint)
    df = df.dropna(subset=['step']).copy()
    df['step'] = df['step'].astype(int)
    df = df.sort_values('step').drop_duplicates('step')
    return df[['step', 'f1']].reset_index(drop=True)


def plot_all_f1(csv_dir=".", output_path="all_languages_f1.pdf"):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    plotted = 0
    for lang_code, lang_label, color, marker in LANGUAGES:
        csv_path = os.path.join(csv_dir, f"{lang_code}.csv")
        if not os.path.isfile(csv_path):
            print(f"  [skip] {csv_path} not found")
            continue

        df = load_lang(csv_path)
        if df.empty:
            continue

        max_step = df['step'].max()
        pct_train = df['step'] / max_step * 100      # 0–100 %

        ax.plot(
            pct_train,
            df['f1'],
            color=color,
            linestyle="-",
            marker=marker,
            markersize=3.0,
            linewidth=1.5,
            alpha=0.90,
            markevery=max(1, len(df) // 8),
            label=lang_label,
        )
        plotted += 1

    if plotted == 0:
        print("No CSV files found. Nothing to plot.")
        return

    # Axes
    ax.set_xlabel("% Pretraining Steps", fontsize=11)
    ax.set_ylabel("Morphological F1 Score", fontsize=11)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%g%%'))
    ax.tick_params(axis='both', labelsize=10)

    # ── Background phase bands ──────────────────────────────────────────────
    PHASES = [
        (0,   2.5,   "#d0e8fb", "Rapid evolution"),        # light blue
        (1.5,   10,  "#fef3cd", "\nFluctuation in\nagglutinative languages"),              # light amber
        (10,  25,  "#d4edda", "Convergence"),              # light green
        (25,  100, "#f8d7da", "Saturation"),               # light red/pink
    ]
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

    # Grid – drawn on top of the bands
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.45, zorder=1)
    ax.set_axisbelow(False)   # grid already at zorder=1, bands at zorder=0

    # Legend — placed above the plot, one row per language
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        fontsize=9,
        frameon=True,
        framealpha=0.95,
        edgecolor='#cccccc',
        columnspacing=1.0,
        handlelength=1.8,
        handletextpad=0.4,
        borderpad=0.5,
    )

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(output_path.replace('.pdf', '.png'), format='png',
                bbox_inches='tight', dpi=300)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    plot_all_f1(csv_dir=SCRIPT_DIR, output_path=os.path.join(SCRIPT_DIR, "all_languages_f1.pdf"))

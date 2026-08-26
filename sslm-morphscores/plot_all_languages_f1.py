import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Language metadata:  file-prefix -> display label
# ---------------------------------------------------------------------------
LANGUAGES = {
    "eng_latn": "English",
    "fas_arab": "Persian",
    "fin_latn": "Finnish",
    "heb_hebr": "Hebrew",
    "hin_deva": "Hindi",
    "hrv_latn": "Croatian",
    "hun_latn": "Hungarian",
    "ind_latn": "Indonesian",
    "kir_cyrl": "Kyrgyz",
    "mal_mlym": "Malayalam",
    "rus_cyrl": "Russian",
    "snd_arab": "Sindhi",
    "swe_latn": "Swedish",
    "tam_taml": "Tamil",
    "tel_telu": "Telugu",
    "tur_latn": "Turkish",
}

# ---------------------------------------------------------------------------
# Distinct visual styles so every line is unambiguous in B&W and colour print
# ---------------------------------------------------------------------------
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94",
]
MARKERS    = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "p",
              "<", ">", "H", "8", "+", "x"]


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
    fig, ax = plt.subplots(figsize=(6.0, 4.0))   # taller to accommodate top legend

    plotted = 0
    for idx, (lang_code, lang_label) in enumerate(LANGUAGES.items()):
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
            color=COLORS[idx % len(COLORS)],
            linestyle="-",
            marker=MARKERS[idx % len(MARKERS)],
            markersize=2.0,
            linewidth=1.1,
            alpha=0.90,
            markevery=max(1, len(df) // 10),        # thin out markers
            label=lang_label,
        )
        plotted += 1

    if plotted == 0:
        print("No CSV files found. Nothing to plot.")
        return

    # Axes
    ax.set_xlabel("% Pretraining Steps", fontsize=10)
    ax.set_ylabel("Morphological F1 Score", fontsize=10)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%g%%'))
    ax.tick_params(axis='both', labelsize=10)

    # Grid
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)

    # Legend — placed above the plot in 4-column rows
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 1.01),
        ncol=6,
        fontsize=10,
        frameon=True,
        framealpha=0.95,
        edgecolor='#cccccc',
        columnspacing=0.7,
        handlelength=1.4,
        handletextpad=0.3,
        borderpad=0.4,
    )

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(output_path.replace('.pdf', '.png'), format='png',
                bbox_inches='tight', dpi=300)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    plot_all_f1(csv_dir=SCRIPT_DIR, output_path=os.path.join(SCRIPT_DIR, "all_languages_f1.pdf"))

#!/usr/bin/env bash
# =============================================================================
# plot_graphs.sh
# Convenience wrapper around plot_graphs.py.
#
# Usage:
#   bash plot_graphs.sh [OPTIONS]
#
# Options:
#   -m | --metric      METRIC        Metric to plot (required).
#                                    One of: fertility, compression, vocab_size,
#                                    ttr, morphscore_f1, morphscore_precision,
#                                    morphscore_recall, renyi_efficiency,
#                                    exponence, length_dist
#                                    May be repeated or space-separated.
#   -l | --languages   LANG...       Space-separated language codes.
#                                    Default: all languages in CSV.
#                                    e.g.  fin hun tam tel hin
#   -t | --tokenizers  TOK...        Space-separated tokenizer keys.
#                                    Default: all tokenizers in CSV.
#                                    e.g.  bpe unigram sslm hnet morfessor
#   -v | --vocab_sizes SIZE...       Space-separated vocab sizes (no "v" prefix).
#                                    Default: all vocab sizes.
#                                    e.g.  10000 20000
#   -i | --input_dir   DIR           CSV input directory (default: evaluation_csv_results)
#   -o | --output_dir  DIR           Plot output directory (default: plots)
#   -s | --figsize     W H           Figure width and height in inches.
#   --title            "TITLE"       Custom plot title.
#   --dpi              N             Image DPI (default: 150).
#   --no_legend                      Omit the legend.
#   --group_by         typology|script  Background color grouping (default: typology).
#   -h | --help                      Show this help and exit.
#
# Examples:
#   # Plot all languages, all tokenizers, fertility metric
#   bash plot_graphs.sh --metric fertility
#
#   # Plot only Dravidian + Finno-Ugric languages, only v10000, morphscore_f1
#   bash plot_graphs.sh -m morphscore_f1 \
#       -l fin hun tam tel mal \
#       -v 10000
#
#   # Compare a small set of tokenizers across all languages
#   bash plot_graphs.sh -m compression \
#       -t bpe unigram sslm hnet morfessor wordpiece \
#       -v 10000 20000
#
#   # Plot multiple metrics in one call
#   bash plot_graphs.sh -m fertility compression ttr \
#       -l fin hun tam tel \
#       -t bpe unigram sslm hnet
# =============================================================================
set -euo pipefail

# ── defaults ──────────────────────────────────────────────────────────────────
METRICS=()
LANGUAGES=(fin hun mal tam tel kir tur mon ind san hin snd hrv rus fas eng swe heb)
TOKENIZERS=()
VOCAB_SIZES=(10000)
INPUT_DIR="evaluation_csv_results"
OUTPUT_DIR="plots"
FIGSIZE=(11 6)
TITLE=""
DPI=150
NO_LEGEND=""
GROUP_BY="typology"

# ── argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--metric)
            shift
            while [[ $# -gt 0 && "$1" != -* ]]; do
                METRICS+=("$1"); shift
            done ;;
        -l|--languages)
            shift
            while [[ $# -gt 0 && "$1" != -* ]]; do
                LANGUAGES+=("$1"); shift
            done ;;
        -t|--tokenizers)
            shift
            while [[ $# -gt 0 && "$1" != -* ]]; do
                TOKENIZERS+=("$1"); shift
            done ;;
        -v|--vocab_sizes)
            shift
            while [[ $# -gt 0 && "$1" != -* ]]; do
                VOCAB_SIZES+=("$1"); shift
            done ;;
        -i|--input_dir)
            INPUT_DIR="$2"; shift 2 ;;
        -o|--output_dir)
            OUTPUT_DIR="$2"; shift 2 ;;
        -s|--figsize)
            FIGSIZE=("$2" "$3"); shift 3 ;;
        --title)
            TITLE="$2"; shift 2 ;;
        --dpi)
            DPI="$2"; shift 2 ;;
        --no_legend)
            NO_LEGEND="--no_legend"; shift ;;
        --group_by)
            GROUP_BY="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^# ===\+$/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "[ERROR] Unknown option: $1" >&2
            exit 1 ;;
    esac
done

# ── validation ────────────────────────────────────────────────────────────────
if [[ ${#METRICS[@]} -eq 0 ]]; then
    echo "[ERROR] At least one --metric is required." >&2
    exit 1
fi

# ── build common python args ──────────────────────────────────────────────────
PY_ARGS=(
    --input_dir  "$INPUT_DIR"
    --output_dir "$OUTPUT_DIR"
    --dpi        "$DPI"
)

[[ ${#LANGUAGES[@]}   -gt 0 ]] && PY_ARGS+=(--languages   "${LANGUAGES[@]}")
[[ ${#TOKENIZERS[@]}  -gt 0 ]] && PY_ARGS+=(--tokenizers  "${TOKENIZERS[@]}")
[[ ${#VOCAB_SIZES[@]} -gt 0 ]] && PY_ARGS+=(--vocab_sizes "${VOCAB_SIZES[@]}")
[[ ${#FIGSIZE[@]}     -gt 0 ]] && PY_ARGS+=(--figsize     "${FIGSIZE[@]}")
[[ -n "$TITLE"               ]] && PY_ARGS+=(--title       "$TITLE")
[[ -n "$NO_LEGEND"           ]] && PY_ARGS+=("$NO_LEGEND")
PY_ARGS+=(--group_by "$GROUP_BY")

# ── run ───────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT" || exit 1  # relative --input_dir/--output_dir defaults resolve from repo root
PYTHON="${PYTHON:-python3}"

echo "============================================================"
echo "  Output directory : $OUTPUT_DIR"
echo "  Metrics          : ${METRICS[*]}"
[[ ${#LANGUAGES[@]}   -gt 0 ]] && echo "  Languages        : ${LANGUAGES[*]}"
[[ ${#TOKENIZERS[@]}  -gt 0 ]] && echo "  Tokenizers       : ${TOKENIZERS[*]}"
[[ ${#VOCAB_SIZES[@]} -gt 0 ]] && echo "  Vocab sizes      : ${VOCAB_SIZES[*]}"
echo "  Group by         : $GROUP_BY"
echo "============================================================"

for METRIC in "${METRICS[@]}"; do
    echo ""
    echo "→ Plotting: $METRIC"
    "$PYTHON" "$SCRIPT_DIR/plot_graphs.py" \
        --metric "$METRIC" \
        "${PY_ARGS[@]}"
done

echo ""
echo "All plots saved in: $OUTPUT_DIR/"

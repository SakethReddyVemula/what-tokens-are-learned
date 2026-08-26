#!/usr/bin/env bash
# =============================================================================
# plot_metric_dynamics.sh
# Convenience wrapper around plot_metric_dynamics.py.
#
# Plots how a tokenization metric evolves across SSLM training checkpoints
# and overlays a horizontal dashed line for H-Nets (best model).
#
# Usage:
#   bash plot_metric_dynamics.sh [OPTIONS]
#
# Options:
#   -m | --metric       METRIC        Metric to plot (required, repeatable).
#                                     One of:
#                                       compression, exponence, fertility,
#                                       mean_token_length, renyi_efficiency, renyi_entropy,
#                                       ttr, vocab_size
#   -l | --languages    LANG...       Space-separated language codes.
#                                     Default: all languages found in metric dir.
#                                     e.g.  fin hun tam tel hin eng
#   -i | --input_dir    DIR           Root evaluation results directory.
#                                     Default: evaluation_results
#   -o | --output_dir   DIR           Plot output directory.
#                                     Default: plots_dynamics
#   -s | --figsize      W H           Figure width and height in inches.
#                                     Default: 8 4.5
#   --title             "TITLE"       Custom plot title.
#   --dpi               N             Image DPI (default: 200).
#   --pct_steps                       X-axis as % of training steps.
#   --no_legend                       Omit the legend.
#   -h | --help                       Show this help and exit.
#
# Examples:
#   # Fertility dynamics, all languages
#   bash plot_metric_dynamics.sh --metric fertility
#
#   # Multiple metrics, subset of languages, % x-axis
#   bash plot_metric_dynamics.sh \
#       --metric fertility ttr compression \
#       --languages eng hin tam fin hun \
#       --pct_steps
#
#   # Mean token length with a custom title
#   bash plot_metric_dynamics.sh \
#       --metric mean_token_length \
#       --title "Mean Token Length Dynamics" \
#       --output_dir plots_dynamics
# =============================================================================
set -euo pipefail

# ── defaults ──────────────────────────────────────────────────────────────────
METRICS=()
LANGUAGES=(eng tam heb hun hin ind)
INPUT_DIR="evaluation_results"
OUTPUT_DIR="plots_dynamics"
FIGSIZE=(8 4.5)
TITLE=""
DPI=200
PCT_STEPS=""
NO_LEGEND=""

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
        --pct_steps)
            PCT_STEPS="--pct_steps"; shift ;;
        --no_legend)
            NO_LEGEND="--no_legend"; shift ;;
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
    echo "        Valid: compression, exponence, fertility, mean_token_length," >&2
    echo "               renyi_entropy, ttr, vocab_size" >&2
    exit 1
fi

# ── build common python args ──────────────────────────────────────────────────
PY_ARGS=(
    --input_dir  "$INPUT_DIR"
    --output_dir "$OUTPUT_DIR"
    --dpi        "$DPI"
    --figsize    "${FIGSIZE[@]}"
)

[[ ${#LANGUAGES[@]} -gt 0 ]] && PY_ARGS+=(--languages "${LANGUAGES[@]}")
[[ -n "$TITLE"            ]] && PY_ARGS+=(--title     "$TITLE")
[[ -n "$PCT_STEPS"        ]] && PY_ARGS+=("$PCT_STEPS")
[[ -n "$NO_LEGEND"        ]] && PY_ARGS+=("$NO_LEGEND")

# ── run ───────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT" || exit 1  # relative --input_dir/--output_dir defaults resolve from repo root
PYTHON="${PYTHON:-python3}"

echo "============================================================"
echo "  Input directory  : $INPUT_DIR"
echo "  Output directory : $OUTPUT_DIR"
echo "  Metrics          : ${METRICS[*]}"
[[ ${#LANGUAGES[@]} -gt 0 ]] && echo "  Languages        : ${LANGUAGES[*]}"
echo "  Figure size      : ${FIGSIZE[*]} inches"
echo "  DPI              : $DPI"
[[ -n "$PCT_STEPS" ]] && echo "  X-axis           : % pretraining steps"
echo "============================================================"

for METRIC in "${METRICS[@]}"; do
    echo ""
    echo "→ Plotting dynamics: $METRIC"
    "$PYTHON" "$SCRIPT_DIR/plot_metric_dynamics.py" \
        --metric "$METRIC" \
        "${PY_ARGS[@]}"
done

echo ""
echo "All plots saved in: $OUTPUT_DIR/"

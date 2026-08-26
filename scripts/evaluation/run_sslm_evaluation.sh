#!/bin/bash

# Resolve paths relative to this script, then run from the repo root so that
# default relative output dirs (tokenizers-bin/, evaluation_results/, ...) resolve there.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT" || exit 1

# Default arguments
LANGS="fin hun mal tam tel kir tur mon ind san hin snd hrv rus fas eng swe heb"
SPLIT="test"
REPO="SakethVemula/sslm-corpus-segments"
# METRICS="fertility compression freq_dist length_dist exponence vocab_size ttr renyi_entropy renyi_efficiency shannon_entropy shannon_efficiency"
METRICS="renyi_efficiency"
# METRICS="fertility compression freq_dist length_dist exponence vocab_size ttr"
OUTPUT_DIR="evaluation_results"
STEP="" # If set, considers the given step number, if not considers the highest (latest) step number and evaluate against it
EVAL_ALL_STEPS="true"


while [[ "$#" -gt 0 ]]; do
    case $1 in
        --langs) LANGS="$2"; shift ;;
        --step) STEP="$2"; shift ;;
        --eval_all_steps) EVAL_ALL_STEPS="true" ;;
        --split) SPLIT="$2"; shift ;;
        --repo) REPO="$2"; shift ;;
        --metrics) METRICS="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$LANGS" ]; then
    echo "Usage: $0 --langs \"lang1 lang2\" [--step \"24\"] [--metrics \"fertility compression\"]"
    exit 1
fi

echo "==================================="
echo "Starting SSLM tokenizer evaluation..."
echo "Languages: $LANGS"
if [ -n "$STEP" ]; then
    echo "Step/Epoch: $STEP"
elif [ "$EVAL_ALL_STEPS" = "true" ]; then
    echo "Step/Epoch: All available steps"
else
    echo "Step/Epoch: Highest available step"
fi
echo "Split: $SPLIT"
echo "Metrics: $METRICS"
echo "Output Directory: $OUTPUT_DIR"
echo "==================================="

CMD="python3 "$SCRIPT_DIR/evaluate_sslm_tokenizer.py" \
    --langs $LANGS \
    --split $SPLIT \
    --hf_repo \"$REPO\" \
    --metrics $METRICS \
    --output_dir $OUTPUT_DIR"

if [ -n "$STEP" ]; then
    CMD="$CMD --step $STEP"
fi

if [ "$EVAL_ALL_STEPS" = "true" ]; then
    CMD="$CMD --eval_all_steps"
fi

eval $CMD

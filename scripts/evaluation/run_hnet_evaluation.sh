#!/bin/bash

# Resolve paths relative to this script, then run from the repo root so that
# default relative output dirs (tokenizers-bin/, evaluation_results/, ...) resolve there.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT" || exit 1

# Default arguments
LANGS="fin hun mal tam tel kir tur mon ind san hin snd hrv rus fas eng swe heb"
SPLIT="test"
REPO="SakethVemula/hnet-segments"
# METRICS="fertility compression freq_dist length_dist exponence vocab_size ttr renyi_entropy renyi_efficiency shannon_entropy shannon_efficiency"
# METRICS="renyi_entropy renyi_efficiency"
METRICS="fertility compression freq_dist length_dist exponence vocab_size ttr renyi_entropy renyi_efficiency"
OUTPUT_DIR="evaluation_results"


while [[ "$#" -gt 0 ]]; do
    case $1 in
        --langs) LANGS="$2"; shift ;;
        --split) SPLIT="$2"; shift ;;
        --repo) REPO="$2"; shift ;;
        --metrics) METRICS="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$LANGS" ]; then
    echo "Usage: $0 --langs \"lang1 lang2\" [--metrics \"fertility compression\"]"
    exit 1
fi

echo "==================================="
echo "Starting H-Net tokenizer evaluation..."
echo "Languages: $LANGS"
echo "Split: $SPLIT"
echo "Metrics: $METRICS"
echo "Output Directory: $OUTPUT_DIR"
echo "==================================="

CMD="python3 "$SCRIPT_DIR/evaluate_hnet_tokenizer.py" \
    --langs $LANGS \
    --split $SPLIT \
    --hf_repo \"$REPO\" \
    --metrics $METRICS \
    --output_dir $OUTPUT_DIR"

eval $CMD

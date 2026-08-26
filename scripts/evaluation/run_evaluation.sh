#!/bin/bash

# Resolve paths relative to this script, then run from the repo root so that
# default relative output dirs (tokenizers-bin/, evaluation_results/, ...) resolve there.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT" || exit 1

# Default arguments
LANGS="fin hun mal tam tel kir tur mon ind san hin snd hrv rus fas eng swe heb"
# LANGS="tel"
TOKENIZERS="superbpe" # bpe / unigram / wordpiece / bpe-dropout / morfessor / superbpe / boundlessbpe / pickybpe / pathpiece / myte / sage / morphbpe / morphulm / morphwp
VOCAB_SIZE=25000 # T (final vocab size)
SUPERBPE_BASE_VOCAB_SIZE=10000 # t (vocab size of subword stage, and the transition point)
SPLIT="test"
REPO="SakethVemula/fixed-tokenizer-segments"
BPE_DROPOUT="false"
BPE_DROPOUT_PROB="0.1"
# METRICS="fertility compression freq_dist length_dist exponence vocab_size ttr renyi_entropy renyi_efficiency shannon_entropy shannon_efficiency"
METRICS="fertility"
OUTPUT_DIR="evaluation_results"


while [[ "$#" -gt 0 ]]; do
    case $1 in
        --langs) LANGS="$2"; shift ;;
        --tokenizers) TOKENIZERS="$2"; shift ;;
        --vocab_size) VOCAB_SIZE="$2"; shift ;;
        --split) SPLIT="$2"; shift ;;
        --repo) REPO="$2"; shift ;;
        --bpe_dropout) BPE_DROPOUT="true" ;;
        --bpe_dropout_prob) BPE_DROPOUT_PROB="$2"; shift ;;
        --superbpe_base_vocab_size) SUPERBPE_BASE_VOCAB_SIZE="$2"; shift ;;
        --metrics) METRICS="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$LANGS" ]; then
    echo "Usage: $0 --langs \"lang1 lang2\" [--tokenizers \"bpe unigram wordpiece\"] [--metrics \"fertility compression\"]"
    exit 1
fi

echo "==================================="
echo "Starting tokenizer evaluation..."
echo "Languages: $LANGS"
echo "Tokenizers: $TOKENIZERS"
echo "Vocab Size: $VOCAB_SIZE"
echo "Split: $SPLIT"
echo "Metrics: $METRICS"
echo "BPE Dropout: $BPE_DROPOUT (Prob: $BPE_DROPOUT_PROB)"
echo "Output Directory: $OUTPUT_DIR"
echo "==================================="

CMD="python3 "$SCRIPT_DIR/evaluate_tokenizers.py" \
    --langs $LANGS \
    --model_types $TOKENIZERS \
    --vocab_size $VOCAB_SIZE \
    --split $SPLIT \
    --hf_repo \"$REPO\" \
    --metrics $METRICS \
    --output_dir $OUTPUT_DIR"

if [ "$BPE_DROPOUT" = "true" ]; then
    CMD="$CMD --bpe_dropout --bpe_dropout_prob $BPE_DROPOUT_PROB"
fi

if [[ "$TOKENIZERS" == *"superbpe"* ]]; then
    CMD="$CMD --superbpe_base_vocab_size $SUPERBPE_BASE_VOCAB_SIZE"
fi

eval $CMD

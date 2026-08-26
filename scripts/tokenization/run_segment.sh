#!/bin/bash

# Resolve paths relative to this script, then run from the repo root so that
# default relative output dirs (tokenizers-bin/, evaluation_results/, ...) resolve there.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT" || exit 1

# Default arguments
# LANGS="fin hun mal tam tel kir tur mon ind san hin snd hrv rus fas eng swe heb"
LANGS="tel"
TOKENIZERS="bpe unigram wordpiece morfessor bpe-dropout pickybpe superbpe boundlessbpe pathpiece myte sage morphbpe morphulm morphwp" # bpe / unigram / wordpiece / bpe-dropout / morfessor / superbpe / boundlessbpe / pickybpe / pathpiece / myte / sage / morphbpe / morphulm / morphwp
VOCAB_SIZE=10000 # T (final vocab size)
SUPERBPE_BASE_VOCAB_SIZE=4000 # t (vocab size of subword stage, and the transition point)
SPLIT="valid"
BATCH_SIZE=5
REPO="SakethVemula/fixed-tokenizer-segments"
BPE_DROPOUT="false"
BPE_DROPOUT_PROB="0.1"


while [[ "$#" -gt 0 ]]; do
    case $1 in
        --langs) LANGS="$2"; shift ;;
        --tokenizers) TOKENIZERS="$2"; shift ;;
        --vocab_size) VOCAB_SIZE="$2"; shift ;;
        --split) SPLIT="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --repo) REPO="$2"; shift ;;
        --bpe_dropout) BPE_DROPOUT="true" ;;
        --bpe_dropout_prob) BPE_DROPOUT_PROB="$2"; shift ;;
        --superbpe_base_vocab_size) SUPERBPE_BASE_VOCAB_SIZE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$LANGS" ]; then
    echo "Usage: $0 --langs \"lang1 lang2\" [--tokenizers \"bpe unigram wordpiece morfessor\"]"
    exit 1
fi

echo "==================================="
echo "Starting segmentation and upload..."
echo "Languages: $LANGS"
echo "Tokenizers: $TOKENIZERS"
echo "Vocab Size: $VOCAB_SIZE"
echo "Split: $SPLIT"
echo "BPE Dropout: $BPE_DROPOUT (Prob: $BPE_DROPOUT_PROB)"
echo "==================================="

CMD="python3 "$SCRIPT_DIR/run_segment.py" \
    --langs $LANGS \
    --model_types $TOKENIZERS \
    --vocab_size $VOCAB_SIZE \
    --split $SPLIT \
    --batch_size $BATCH_SIZE \
    --hf_repo \"$REPO\""

if [ "$BPE_DROPOUT" = "true" ]; then
    CMD="$CMD --bpe_dropout --bpe_dropout_prob $BPE_DROPOUT_PROB"
fi

if [[ "$TOKENIZERS" == *"superbpe"* ]]; then
    CMD="$CMD --superbpe_base_vocab_size $SUPERBPE_BASE_VOCAB_SIZE"
fi

eval $CMD

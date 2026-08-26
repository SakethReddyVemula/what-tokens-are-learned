#!/bin/bash

# Resolve paths relative to this script, then run from the repo root so that
# default relative output dirs (tokenizers-bin/, evaluation_results/, ...) resolve there.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT" || exit 1

# Configuration
LANGS="fin hun mal tam tel kir tur ind hin snd hrv rus fas eng swe heb"
VOCAB_SIZES="10000"
MODEL_TYPES="boundlessbpe"  # bpe / unigram / wordpiece / bpe-dropout (set true and run with bpe) / morfessor / superbpe / boundlessbpe / pickybpe / pathpiece / myte / morphbpe / morphulm / morphwp

# SuperBPE Configuration
SUPERBPE_BASE_VOCAB_SIZE=4000
SUPERBPE_STAGE1_REGEX="[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
SUPERBPE_STAGE2_REGEX="\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)"

# BoundlessBPE Configuration
BOUNDLESSBPE_TAU=0.9
BOUNDLESSBPE_RECALC=1000
BOUNDLESSBPE_PATNAME="ultimate2"
BOUNDLESSBPE_BLOWUP=1

# PickyBPE Configuration
PICKYBPE_THRESHOLD=0.9

# BPE Dropout Configuration
BPE_DROPOUT="false"
BPE_DROPOUT_PROB=0.1

# Hugging Face Configuration
HF_REPO="SakethVemula/fixed-tokenizer-morphscore-segments"

SCRIPT="$SCRIPT_DIR/run_morphscore.py"

if [ ! -f "$SCRIPT" ]; then
    echo "Error: Python script '$SCRIPT' not found in current directory."
    exit 1
fi

echo "Starting morphscore evaluation pipeline..."
echo "  Languages: $LANGS"
echo "  Vocab Sizes: $VOCAB_SIZES"
echo "  Model Types: $MODEL_TYPES"
echo "  HF Repo: $HF_REPO"

CMD="python3 $SCRIPT --langs $LANGS --vocab_sizes $VOCAB_SIZES --model_types $MODEL_TYPES --hf_repo $HF_REPO"

if [ -n "$HF_TOKEN" ]; then
    CMD="$CMD --hf_token $HF_TOKEN"
fi

if [ "$BPE_DROPOUT" = "true" ]; then
    CMD="$CMD --bpe_dropout --bpe_dropout_prob $BPE_DROPOUT_PROB"
fi

if [[ "$MODEL_TYPES" == *"superbpe"* ]]; then
    CMD="$CMD --superbpe_base_vocab_size $SUPERBPE_BASE_VOCAB_SIZE"
fi

# Execute
echo "Running: $CMD"
eval "$CMD"

if [ $? -eq 0 ]; then
    echo "Morphscore pipeline completed successfully."
else
    echo "Morphscore pipeline failed."
    exit 1
fi

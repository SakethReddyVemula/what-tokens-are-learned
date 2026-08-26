#!/bin/bash

# Resolve paths relative to this script, then run from the repo root so that
# default relative output dirs (tokenizers-bin/, evaluation_results/, ...) resolve there.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT" || exit 1

# Configuration
LANG="eng"
SPLIT="valid"
VOCAB_SIZE=8000
MODEL_TYPE="bpe"

# Optional overrides
INPUT_FILE=""
OUTPUT_FILE=""

# Script path
SCRIPT="$SCRIPT_DIR/apply_tokenizer.py"

if [ ! -f "$SCRIPT" ]; then
    echo "Error: Python script '$SCRIPT' not found in current directory."
    exit 1
fi

echo "Applying tokenizer for language: $LANG, split: $SPLIT, vocab: $VOCAB_SIZE"

CMD="python3 $SCRIPT --lang $LANG --split $SPLIT --vocab_size $VOCAB_SIZE --model_type $MODEL_TYPE"

if [ -n "$INPUT_FILE" ]; then
    CMD="$CMD --input_file $INPUT_FILE"
fi

if [ -n "$OUTPUT_FILE" ]; then
    CMD="$CMD --output_file $OUTPUT_FILE"
fi

# Execute
echo "Running: $CMD"
$CMD

if [ $? -eq 0 ]; then
    echo "Processing completed successfully."
else
    echo "Processing failed."
    exit 1
fi

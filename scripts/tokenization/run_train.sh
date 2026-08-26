#!/bin/bash

# Resolve paths relative to this script, then run from the repo root so that
# default relative output dirs (tokenizers-bin/, evaluation_results/, ...) resolve there.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT" || exit 1

# Configuration
LANG="eng"
VOCAB_SIZE=10000
MODEL_TYPE="bpe"
CHARACTER_COVERAGE=1.0

# Optional: Override input file path if needed. leave empty for default logic in python script.
INPUT_FILE="" 

# Output directory
OUTPUT_DIR="tokenizers-bin"

# Script path
SCRIPT="$SCRIPT_DIR/train_tokenizer.py"

if [ ! -f "$SCRIPT" ]; then
    echo "Error: Python script '$SCRIPT' not found in current directory."
    exit 1
fi

echo "Starting tokenizer training for language: $LANG"
echo "  Vocab Size: $VOCAB_SIZE"
echo "  Model Type: $MODEL_TYPE"

# Build command
CMD="python3 $SCRIPT --lang $LANG --vocab_size $VOCAB_SIZE --model_type $MODEL_TYPE --output_dir $OUTPUT_DIR --character_coverage $CHARACTER_COVERAGE"

if [ -n "$INPUT_FILE" ]; then
    CMD="$CMD --input_file $INPUT_FILE"
fi

# Execute
echo "Running: $CMD"
$CMD

if [ $? -eq 0 ]; then
    echo "Training completed successfully."
else
    echo "Training failed."
    exit 1
fi

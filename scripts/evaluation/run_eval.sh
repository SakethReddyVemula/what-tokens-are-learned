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
MODEL_TYPE="unigram"

# Path to the tokenizer-analysis-suite repository
# Adjust this if your repo is cloned elsewhere
ANALYSIS_SUITE="../tokenizer-analysis-suite"

# Optional overrides
MODEL_PATH=""
INPUT_FILE=""
OUTPUT_DIR=""

# Script path
SCRIPT="$SCRIPT_DIR/evaluate_tokenizer.py"

if [ ! -f "$SCRIPT" ]; then
    echo "Error: Python script '$SCRIPT' not found in current directory."
    exit 1
fi

echo "Evaluating tokenizer for language: $LANG, split: $SPLIT, vocab: $VOCAB_SIZE"

CMD="python3 $SCRIPT --lang $LANG --split $SPLIT --vocab_size $VOCAB_SIZE --model_type $MODEL_TYPE --analysis_suite_path $ANALYSIS_SUITE"

if [ -n "$MODEL_PATH" ]; then
    CMD="$CMD --model_path $MODEL_PATH"
fi

if [ -n "$INPUT_FILE" ]; then
    CMD="$CMD --input_file $INPUT_FILE"
fi

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output_dir $OUTPUT_DIR"
fi

# Execute
echo "Running: $CMD"
$CMD

if [ $? -eq 0 ]; then
    echo "Evaluation completed successfully."
else
    echo "Evaluation failed."
    exit 1
fi

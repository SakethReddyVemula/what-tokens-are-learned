#!/bin/bash

# Resolve paths relative to this script, then run from the repo root so that
# default relative output dirs (tokenizers-bin/, evaluation_results/, ...) resolve there.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT" || exit 1

# Configuration
# Space-separated lists
# LANGS="fin hun mal tam tel kir tur mon ind san hin snd hrv rus fas eng swe heb" 
LANGS="eng"
# LANGS="tel kir tur mon ind san hin snd hrv rus fas eng swe heb" 
# LANGS="fin"
# LANGS="san hin snd hrv rus fas eng swe heb"
# LANGS="kat kor gle ita spa ell lav isl afr fin hun mal tam tel kir tur mon ind san hin snd hrv rus fas eng swe heb"
VOCAB_SIZES="5000 10000 20000" # T (final vocab size)
MODEL_TYPES="morphbpe morphulm morphwp" # bpe / unigram / wordpiece / bpe-dropout / morfessor / superbpe / boundlessbpe / pickybpe / pathpiece / myte / morphbpe / morphulm / morphwp / sslm-bpe / sslm-ulm / sslm-wp

# SuperBPE Configuration
SUPERBPE_BASE_VOCAB_SIZE=4000 # t (vocab size of subword stage, and the transition point)
SUPERBPE_STAGE1_REGEX="[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
SUPERBPE_STAGE2_REGEX="\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)"

# BoundlessBPE Configuration
BOUNDLESSBPE_TAU=0.9
BOUNDLESSBPE_RECALC=1000
BOUNDLESSBPE_PATNAME="ultimate2"
BOUNDLESSBPE_BLOWUP=1

# PickyBPE Configuration
PICKYBPE_THRESHOLD=0.9

DATASET_SPLIT="valid"

# Path to the tokenizer-analysis-suite repository
ANALYSIS_SUITE="../tokenizer-analysis-suite"

# MorphScore Configuration
MORPHSCORE="false" # Set to "true" to enable
MORPHSCORE_DATA_DIR="morphscore_data"

# Script path
SCRIPT="$SCRIPT_DIR/run_pipeline.py"

if [ ! -f "$SCRIPT" ]; then
    echo "Error: Python script '$SCRIPT' not found in current directory."
    exit 1
fi

echo "Starting pipeline..."
echo "  Languages: $LANGS"
echo "  Vocab Sizes: $VOCAB_SIZES"
echo "  Model Types: $MODEL_TYPES"
echo "  Split: $DATASET_SPLIT"
echo "  MorphScore: $MORPHSCORE"

CMD="python3 $SCRIPT --langs $LANGS --vocab_sizes $VOCAB_SIZES --model_types $MODEL_TYPES --dataset_split $DATASET_SPLIT --analysis_suite_path $ANALYSIS_SUITE"

if [ "$MORPHSCORE" = "true" ]; then
    CMD="$CMD --morphscore --morphscore_data_dir $MORPHSCORE_DATA_DIR"
fi

if [[ "$MODEL_TYPES" == *"superbpe"* ]]; then
    CMD="$CMD --superbpe_base_vocab_size $SUPERBPE_BASE_VOCAB_SIZE --superbpe_stage1_regex \"$SUPERBPE_STAGE1_REGEX\" --superbpe_stage2_regex \"$SUPERBPE_STAGE2_REGEX\""
fi

if [[ "$MODEL_TYPES" == *"boundlessbpe"* ]]; then
    CMD="$CMD --boundlessbpe_tau $BOUNDLESSBPE_TAU --boundlessbpe_recalc $BOUNDLESSBPE_RECALC --boundlessbpe_patname \"$BOUNDLESSBPE_PATNAME\" --boundlessbpe_blowup $BOUNDLESSBPE_BLOWUP"
fi

if [[ "$MODEL_TYPES" == *"pickybpe"* ]]; then
    CMD="$CMD --pickybpe_threshold $PICKYBPE_THRESHOLD"
fi

# Execute
echo "Running: $CMD"
eval "$CMD"

if [ $? -eq 0 ]; then
    echo "Pipeline completed successfully."
else
    echo "Pipeline failed."
    exit 1
fi

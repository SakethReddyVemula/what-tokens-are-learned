#!/bin/bash

#SBATCH --job-name=boundlessbpe
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=2048
#SBATCH --time=3-00:00:00
#SBATCH --output=boundlessbpe_pretraning.txt
#SBATCH --mail-user=saketh.vemula@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --nodelist=gnode067

# Resolve paths relative to this script, then run from the repo root so that
# default relative output dirs (tokenizers-bin/, evaluation_results/, ...) resolve there.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT" || exit 1

source /home2/$USER/sslm-venv/bin/activate

# Configuration
# Space-separated lists
LANGS="fin hun mal tam tel kir tur mon ind san hin snd hrv rus fas eng swe heb" 
# LANGS="eng"
# LANGS="kat kor gle ita spa ell lav isl afr fin hun mal tam tel kir tur mon ind san hin snd hrv rus fas eng swe heb"
VOCAB_SIZES="10000" # T (final vocab size)
MODEL_TYPES="myte" # bpe / unigram / wordpiece / bpe-dropout / morfessor / superbpe / boundlessbpe / myte

# SuperBPE Configuration
SUPERBPE_BASE_VOCAB_SIZE=10000 # t (vocab size of subword stage, and the transition point)
SUPERBPE_STAGE1_REGEX="[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
SUPERBPE_STAGE2_REGEX="\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)"

# BoundlessBPE Configuration
BOUNDLESSBPE_TAU=0.9
BOUNDLESSBPE_RECALC=1000
BOUNDLESSBPE_PATNAME="ultimate2"
BOUNDLESSBPE_BLOWUP=1

DATASET_SPLIT="valid"

# Path to the tokenizer-analysis-suite repository
ANALYSIS_SUITE="../tokenizer-analysis-suite"

# MorphScore Configuration
MORPHSCORE="false" # Set to "true" to enable
MORPHSCORE_DATA_DIR="morphscore_data"

# Script path
SCRIPT="$SCRIPT_DIR/../run_pipeline.py"

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

# Execute
echo "Running: $CMD"
eval "$CMD"

if [ $? -eq 0 ]; then
    echo "Pipeline completed successfully."
else
    echo "Pipeline failed."
    exit 1
fi

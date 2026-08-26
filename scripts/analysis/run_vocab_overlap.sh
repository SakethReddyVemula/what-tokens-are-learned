#!/bin/bash

# Resolve paths relative to this script, then run from the repo root so that
# default relative output dirs (tokenizers-bin/, evaluation_results/, ...) resolve there.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT" || exit 1

# run_vocab_overlap.sh
# --------------------
# Computes pairwise vocabulary overlap (Jaccard) between segmented outputs
# of different tokenizers, averaged across all requested languages, and saves
# a lower-triangular heatmap (PDF + PNG) to the output directory.

# ── defaults ──────────────────────────────────────────────────────────────────
LANGS="fin hun mal tam tel kir tur mon ind san hin snd hrv rus fas eng swe heb"
# LANGS="fin"
TOKENIZERS="bpe unigram wordpiece morfessor bpe-dropout pickybpe superbpe boundlessbpe pathpiece myte sage morphbpe morphulm morphwp"
VOCAB_SIZE=10000
SUPERBPE_BASE_VOCAB_SIZE=4000
SPLIT="test"

FIXED_REPO="SakethVemula/fixed-tokenizer-segments"
SSLM_REPO="SakethVemula/sslm-corpus-segments"
HNET_REPO="SakethVemula/hnet-segments"

INCLUDE_SSLM="true"
INCLUDE_HNET="true"
BPE_DROPOUT="false"
BPE_DROPOUT_PROB="0.1"
PER_LANG="false"
GROUP_BY="typology"  # typology | script | both
NO_TITLE="false"
DRY_RUN="false"
COMPUTE_STDEV="true"
COMBINED="true"
DPI=200
OUTPUT_DIR="plots_combined"


# ── argument parsing ──────────────────────────────────────────────────────────
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --langs)                  LANGS="$2"; shift ;;
        --tokenizers)             TOKENIZERS="$2"; shift ;;
        --vocab_size)             VOCAB_SIZE="$2"; shift ;;
        --split)                  SPLIT="$2"; shift ;;
        --fixed_repo)             FIXED_REPO="$2"; shift ;;
        --sslm_repo)              SSLM_REPO="$2"; shift ;;
        --hnet_repo)              HNET_REPO="$2"; shift ;;
        --superbpe_base_vocab_size) SUPERBPE_BASE_VOCAB_SIZE="$2"; shift ;;
        --include_sslm)           INCLUDE_SSLM="true" ;;
        --include_hnet)           INCLUDE_HNET="true" ;;
        --bpe_dropout)            BPE_DROPOUT="true" ;;
        --bpe_dropout_prob)       BPE_DROPOUT_PROB="$2"; shift ;;
        --per_lang)               PER_LANG="true" ;;
        --group_by)               GROUP_BY="$2"; shift ;;
        --no_title)               NO_TITLE="true" ;;
        --dry_run)                DRY_RUN="true" ;;
        --compute_stdev)          COMPUTE_STDEV="true" ;;
        --combined)               COMBINED="true" ;;
        --dpi)                    DPI="$2"; shift ;;
        --output_dir)             OUTPUT_DIR="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$LANGS" ]; then
    echo "Usage: $0 --langs \"fin hun ...\" [options]"
    echo ""
    echo "Options:"
    echo "  --langs                  Space-separated language codes (default: all 18)"
    echo "  --tokenizers             Space-separated tokenizer keys (default: all fixed)"
    echo "  --vocab_size             Vocabulary size (default: 25000)"
    echo "  --split                  Corpus split (default: test)"
    echo "  --fixed_repo             HF repo for fixed-tokenizer segments"
    echo "  --sslm_repo              HF repo for SSLM segments"
    echo "  --hnet_repo              HF repo for H-Net segments"
    echo "  --superbpe_base_vocab_size  SuperBPE base vocab size (default: 10000)"
    echo "  --include_sslm           Include SSLM in the comparison"
    echo "  --include_hnet           Include H-Nets in the comparison"
    echo "  --bpe_dropout            Use BPE-Dropout variant"
    echo "  --bpe_dropout_prob       BPE-Dropout probability (default: 0.1)"
    echo "  --per_lang               Also save one heatmap per language"
    echo "  --group_by               Group for group-averaged plots: typology (default) | script | both"
    echo "  --no_title               Omit the plot title"
    echo "  --dry_run                Use synthetic data (no HF downloads)"
    echo "  --compute_stdev          Also plot std-dev heatmaps (variation across languages per group)"
    echo "  --dpi                    Plot resolution (default: 200)"
    echo "  --output_dir             Output directory (default: plots)"
    exit 1
fi

# ── summary ───────────────────────────────────────────────────────────────────
echo "==================================================="
echo "  Pairwise Vocabulary Overlap"
echo "==================================================="
echo "  Languages  : $LANGS"
echo "  Tokenizers : $TOKENIZERS"
echo "  Vocab size : $VOCAB_SIZE"
echo "  Split      : $SPLIT"
echo "  Fixed repo : $FIXED_REPO"
if [ "$INCLUDE_SSLM" = "true" ]; then
    echo "  SSLM repo  : $SSLM_REPO"
fi
if [ "$INCLUDE_HNET" = "true" ]; then
    echo "  H-Net repo : $HNET_REPO"
fi
echo "  BPE Dropout: $BPE_DROPOUT (prob=$BPE_DROPOUT_PROB)"
echo "  Per-lang   : $PER_LANG"
echo "  Group by   : $GROUP_BY"
echo "  Dry run    : $DRY_RUN"
echo "  Stdev plots: $COMPUTE_STDEV"
echo "  Output dir : $OUTPUT_DIR"
echo "==================================================="

# ── build command ─────────────────────────────────────────────────────────────
CMD="python3 "$SCRIPT_DIR/compute_vocab_overlap.py" \
    --langs $LANGS \
    --tokenizers $TOKENIZERS \
    --vocab_size $VOCAB_SIZE \
    --split $SPLIT \
    --fixed_repo \"$FIXED_REPO\" \
    --sslm_repo \"$SSLM_REPO\" \
    --hnet_repo \"$HNET_REPO\" \
    --superbpe_base_vocab_size $SUPERBPE_BASE_VOCAB_SIZE \
    --dpi $DPI \
    --output_dir $OUTPUT_DIR"

[ "$INCLUDE_SSLM" = "true" ]   && CMD="$CMD --include_sslm"
[ "$INCLUDE_HNET" = "true" ]   && CMD="$CMD --include_hnet"
[ "$BPE_DROPOUT"  = "true" ]   && CMD="$CMD --bpe_dropout --bpe_dropout_prob $BPE_DROPOUT_PROB"
[ "$PER_LANG"     = "true" ]   && CMD="$CMD --per_lang"
[ -n "$GROUP_BY" ]             && CMD="$CMD --group_by $GROUP_BY"
[ "$NO_TITLE"     = "true" ]   && CMD="$CMD --no_title"
[ "$DRY_RUN"      = "true" ]   && CMD="$CMD --dry_run"
[ "$COMPUTE_STDEV" = "true" ]  && CMD="$CMD --compute_stdev"
[ "$COMBINED"      = "true" ]  && CMD="$CMD --combined"

eval $CMD

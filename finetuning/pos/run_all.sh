#!/bin/bash
#SBATCH --job-name=pos_all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=1-00:00:00
#SBATCH --output=pos-finetuning-all.out
#SBATCH --error=pos-finetuning-all.err
#SBATCH --mail-user=saketh.vemula@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --nodelist=gnode078

# Resolve the Python entrypoint relative to this script so the job can be
# submitted from any directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1


source /home2/saketh.vemula/bert-venv/bin/activate

get_free_port() {
    python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); port = s.getsockname()[1]; s.close(); print(port)"
}
export MASTER_PORT=$(get_free_port)
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

export WANDB_PROJECT="p3-pos-finetuning"

export HF_HOME="/scratch/saketh_vemula/cache/huggingface"
mkdir -p $HF_HOME

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

declare -A DATASET_CONFIG=(
    ["hin"]="hi_hdtb"
    ["tel"]="te_mtg"
    ["eng"]="en_ewt"
)

LANGUAGES=("hin" "tel" "eng")

MODEL_TYPES=(
    "bpe"
    "unigram"
    "wordpiece"
    "sage"
    "bpe-dropout"
    "pathpiece"
    "pickybpe"
    "morphbpe"
    "morphulm"
    "morphwp"
    "boundlessbpe"
    "superbpe"
    "sslm-bpe"
    "sslm-ulm"
    "sslm-wp"
)

SEEDS=(42 43 44)

VOCAB_SIZE=10000
SUPERBPE_BASE_VOCAB_SIZE=4000

# ---------------------------------------------------------------------------
# Sweep loop: language × tokenizer × seed
# ---------------------------------------------------------------------------

TOTAL=$(( ${#LANGUAGES[@]} * ${#MODEL_TYPES[@]} * ${#SEEDS[@]} ))
COUNT=0

for LANG in "${LANGUAGES[@]}"; do
    DATASET_CFG="${DATASET_CONFIG[$LANG]}"

    for MODEL_TYPE in "${MODEL_TYPES[@]}"; do

        # Handle bpe-dropout alias (same as in run.sh)
        ACTUAL_MODEL_TYPE=$MODEL_TYPE
        BPE_DROPOUT="False"
        if [ "$MODEL_TYPE" == "bpe-dropout" ]; then
            ACTUAL_MODEL_TYPE="bpe"
            BPE_DROPOUT="True"
        fi

        for SEED in "${SEEDS[@]}"; do
            COUNT=$((COUNT + 1))

            # Build job name
            if [ "$MODEL_TYPE" == "superbpe" ]; then
                JOB_NAME="pos_${LANG}_${DATASET_CFG}_${MODEL_TYPE}_${SUPERBPE_BASE_VOCAB_SIZE}_${VOCAB_SIZE}_seed${SEED}"
            else
                JOB_NAME="pos_${LANG}_${DATASET_CFG}_${MODEL_TYPE}_${VOCAB_SIZE}_seed${SEED}"
            fi

            OUTPUT_DIR="./experiments/${JOB_NAME}"
            mkdir -p "$OUTPUT_DIR"

            echo "========================================================"
            echo "[$COUNT/$TOTAL] Starting: $JOB_NAME"
            echo "  Language   : $LANG ($DATASET_CFG)"
            echo "  Model type : $MODEL_TYPE"
            echo "  Seed       : $SEED"
            echo "  Started at : $(date)"
            echo "========================================================"

            # Refresh port for each run
            export MASTER_PORT=$(get_free_port)

            torchrun \
                --master_port $MASTER_PORT \
                --nproc_per_node 1 \
                --nnodes 1 \
            finetune.py \
                --language ${LANG} \
                --dataset_config ${DATASET_CFG} \
                --model_type ${ACTUAL_MODEL_TYPE} \
                --vocab_size ${VOCAB_SIZE} \
                --bpe_dropout ${BPE_DROPOUT} \
                --superbpe_base_vocab_size ${SUPERBPE_BASE_VOCAB_SIZE} \
                --tokenizer_dir ../../tokenizers-bin \
                --max_length 128 \
                --output_dir ${OUTPUT_DIR} \
                --per_device_train_batch_size 32 \
                --per_device_eval_batch_size 32 \
                --num_train_epochs 10 \
                --learning_rate 2e-5 \
                --weight_decay 0.01 \
                --warmup_ratio 0.10 \
                --logging_dir ./log/${JOB_NAME} \
                --logging_steps 50 \
                --evaluation_strategy epoch \
                --save_strategy epoch \
                --save_total_limit 2 \
                --metric_for_best_model f1 \
                --load_best_model_at_end True \
                --greater_is_better True \
                --report_to wandb \
                --run_name ${JOB_NAME} \
                --seed ${SEED} \
                --fp16

            echo "[$COUNT/$TOTAL] Finished: $JOB_NAME at $(date)"
            echo ""
            rm -r ./experiments
            rm -r ./wandb

        done  # seeds
    done  # model types
done  # languages

echo "========================================================"
echo "All $TOTAL runs completed at $(date)"
echo "========================================================"

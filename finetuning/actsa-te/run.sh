#!/bin/bash
#SBATCH --job-name=actsa-te
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=1-00:00:00
#SBATCH --output=actsa-te-finetuning.out
#SBATCH --error=actsa-te-finetuning.err
#SBATCH --mail-user=saketh.vemula@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --nodelist=gnode047

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

export WANDB_PROJECT="p3-actsa-te-finetuning"

export HF_HOME="/scratch/saketh_vemula/cache/huggingface"
mkdir -p $HF_HOME

LANG="tel"
MODEL_TYPE="bpe"
VOCAB_SIZE=10000
BPE_DROPOUT="False"
SUPERBPE_BASE_VOCAB_SIZE=4000

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --lang) LANG="$2"; shift ;;
        --model_type) MODEL_TYPE="$2"; shift ;;
        --vocab_size) VOCAB_SIZE="$2"; shift ;;
        --bpe_dropout) BPE_DROPOUT="$2"; shift ;;
        --superbpe_base_vocab_size) SUPERBPE_BASE_VOCAB_SIZE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

JOB_NAME="actsa_${LANG}_${MODEL_TYPE}_${VOCAB_SIZE}"
if [ "$MODEL_TYPE" == "superbpe" ]; then
    JOB_NAME="actsa_${LANG}_${MODEL_TYPE}_${SUPERBPE_BASE_VOCAB_SIZE}_${VOCAB_SIZE}"
fi

OUTPUT_DIR="./experiments/${JOB_NAME}"
mkdir -p $OUTPUT_DIR

ACTUAL_MODEL_TYPE=$MODEL_TYPE
if [ "$MODEL_TYPE" == "bpe-dropout" ]; then
    ACTUAL_MODEL_TYPE="bpe"
    BPE_DROPOUT="True"
fi

echo "Starting fine-tuning for $JOB_NAME"

torchrun \
    --master_port $MASTER_PORT \
    --nproc_per_node 1 \
    --nnodes 1 \
finetune.py \
    --language ${LANG} \
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
    --fp16

echo "Fine-tuning completed at $(date)"

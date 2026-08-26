#!/bin/bash
#SBATCH --job-name=tel_sslm-wp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=2-00:00:00
#SBATCH --output=tel-bert-pretraining.out
#SBATCH --error=tel-bert-pretraining.err
#SBATCH --mail-user=saketh.vemula@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --nodelist=gnode076

# Resolve the Python entrypoint relative to this script so the job can be
# submitted from any directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1


# Activate virtual environment
source /home2/saketh.vemula/bert-venv/bin/activate

# Parse arguments
LANG="tel"
MODEL_TYPE="sslm-wp" # options: bpe, unigram, wordpiece, sage, bpe-dropout, pathpiece, pickybpe, boundlessbpe, superbpe
VOCAB_SIZE=10000
SUPERBPE_BASE_VOCAB_SIZE=4000

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --lang) LANG="$2"; shift ;;
        --model_type) MODEL_TYPE="$2"; shift ;;
        --vocab_size) VOCAB_SIZE="$2"; shift ;;
        --superbpe_base_vocab_size) SUPERBPE_BASE_VOCAB_SIZE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

JOB_NAME="${LANG}_${MODEL_TYPE}_${VOCAB_SIZE}"
if [ "$MODEL_TYPE" == "superbpe" ]; then
    JOB_NAME="${LANG}_${MODEL_TYPE}_${SUPERBPE_BASE_VOCAB_SIZE}_${VOCAB_SIZE}"
fi
echo "Running Pretraining for $JOB_NAME"

ACTUAL_MODEL_TYPE=$MODEL_TYPE
BPE_DROPOUT="False"
if [ "$MODEL_TYPE" == "bpe-dropout" ]; then
    ACTUAL_MODEL_TYPE="bpe"
    BPE_DROPOUT="True"
fi

get_free_port() {
    python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); port = s.getsockname()[1]; s.close(); print(port)"
}

export MASTER_PORT=$(get_free_port)
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

# APIs & Tokens
export WANDB_PROJECT="bert-pretraining-p3-25M"

# Caching and local node setup
rm -r /scratch/saketh_vemula
mkdir -p /scratch/saketh_vemula/log
mkdir -p /scratch/saketh_vemula/weights
export HF_HOME=/scratch/saketh_vemula/cache/huggingface
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export TRANSFORMERS_CACHE=${HF_HOME}/models

cp -r ~/TokTrainSuite/data /scratch/saketh_vemula/

rm -r ~/.cache

torchrun \
    --master_port $MASTER_PORT \
    --nproc_per_node 4 \
    --nnodes 1 \
    pretrain_25M.py \
    --language $LANG \
    --model_type $ACTUAL_MODEL_TYPE \
    --vocab_size $VOCAB_SIZE \
    --bpe_dropout $BPE_DROPOUT \
    --superbpe_base_vocab_size $SUPERBPE_BASE_VOCAB_SIZE \
    --train_data_path /scratch/saketh_vemula/data/${LANG}/train.${LANG} \
    --eval_data_path /scratch/saketh_vemula/data/${LANG}/valid.${LANG} \
    --tokenizer_dir ../tokenizers-bin \
    --output_dir /scratch/saketh_vemula/weights/${JOB_NAME} \
    --logging_dir /scratch/saketh_vemula/log/${JOB_NAME} \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --adam_epsilon 1e-6 \
    --max_grad_norm 0.5 \
    --num_train_epochs 5 \
    --warmup_steps 3750 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 2 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --prediction_loss_only True \
    --seed=42 \
    --per_device_train_batch_size 32 \
    --logging_steps 100 \
    --disable_tqdm False \
    --fp16

rm -r ~/.cache


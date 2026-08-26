#!/bin/bash
#SBATCH --job-name=tel-berts
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2048
#SBATCH --time=3-00:00:00
#SBATCH --output=tel-bert-pretraining.out
#SBATCH --error=tel-bert-pretraining.err
#SBATCH --mail-user=saketh.vemula@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --nodelist=gnode075

# Resolve the Python entrypoint relative to this script so the job can be
# submitted from any directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1


# Define your target grids here
LANGUAGES=("tel")  # Add or remove languages as needed
MODEL_TYPES=("bpe" "unigram" "wordpiece" "sage" "bpe-dropout" "pathpiece" "pickybpe" "boundlessbpe" "superbpe")
VOCAB_SIZE=10000

echo "Starting sequential SLURM job sweep..."

# Loop over all requested combinations
for lang in "${LANGUAGES[@]}"; do
    for mt in "${MODEL_TYPES[@]}"; do
    
        echo "------------------------------------------------------"
        echo "Running Language: $lang | Model Type: $mt"
        
        # We invoke bash directly on your existing pretrain.sh script.
        # Because THIS master script is submitted via sbatch, it already has the GPU allocations.
        # Calling bash pretrain.sh executes it sequentially on those allocated GPUs.
        # || true ensures a single tokenizer failure doesn't crash the entire multi-day sweep.
        bash pretrain.sh \
            --lang "$lang" \
            --model_type "$mt" \
            --vocab_size "$VOCAB_SIZE" \
            || { echo "WARNING: Failed to run $lang $mt. Continuing..."; true; }
            
    done
done

echo "------------------------------------------------------"
echo "All grid combinations processed!"


#!/bin/bash
#SBATCH --job-name=qa_inference
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err
#SBATCH --time=02:00:00              # Inference only, shorter time (2 hours)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# GPU selection: Use GPU_ID if set, otherwise use SLURM's assigned GPU
if [ -n "$GPU_ID" ]; then
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    echo "Using GPU: $GPU_ID (from GPU_ID environment variable)"
else
    export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
    echo "Using GPU: $SLURM_LOCALID (assigned by SLURM)"
fi

# Create log directory
mkdir -p logs

# Change to submit directory
cd "$SLURM_SUBMIT_DIR"

# Project root (submit dir). Override with PROJECT_ROOT env if needed.
PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Disable tokenizers parallelism warning (safe for SLURM/forked processes)
export TOKENIZERS_PARALLELISM=false

# Run inference (interactive mode or with predefined questions)
# Set PEFT_MODEL and BASE_MODEL environment variables before submitting
python "$PROJECT_ROOT/src/inference/inference.py" \
    --base-model "${BASE_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}" \
    --peft-model "${PEFT_MODEL:-fine_tuned_weights}" \
    ${QUESTION:+--question "$QUESTION"}

echo "Inference job completed at $(date)"


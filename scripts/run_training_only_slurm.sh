#!/bin/bash
#SBATCH --job-name=qa_training
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err
#SBATCH --time=12:00:00              # Training only, shorter time
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1                 # GPU usage
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

# Run pipeline with skip-generation flag (assumes dataset already exists)
bash "$PROJECT_ROOT/scripts/run_pipeline.sh" \
    --skip-generation \
    --model "${MODEL_NAME:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}" \
    --output-dir "${OUTPUT_DIR:-data_output}" \
    --output-model-dir "${OUTPUT_MODEL_DIR:-fine_tuned_weights}" \
    --epochs "${NUM_EPOCHS:-10}" \
    --batch-size "${BATCH_SIZE:-1}"

echo "Training job completed at $(date)"


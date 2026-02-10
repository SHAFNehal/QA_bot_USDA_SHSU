#!/bin/bash
#SBATCH --job-name=qa_pipeline_cpu
#SBATCH --output=logs/pipeline_cpu_%j.out
#SBATCH --error=logs/pipeline_cpu_%j.err
#SBATCH --time=48:00:00              # CPU is slower, longer timeout
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16           # More cores for CPU
#SBATCH --mem=64G                    # More RAM for CPU
#SBATCH --partition=cpu              # CPU partition (change according to your cluster)

# Create log directory
mkdir -p logs

# Change to submit directory
cd "$SLURM_SUBMIT_DIR"

# Project root (submit dir). Override with PROJECT_ROOT env if needed.
PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Disable tokenizers parallelism warning (safe for SLURM/forked processes)
export TOKENIZERS_PARALLELISM=false

# Run pipeline in CPU mode with all parameters
bash "$PROJECT_ROOT/scripts/run_pipeline.sh" \
    --model "${MODEL_NAME:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}" \
    --input-dir "${INPUT_DIR:-$PROJECT_ROOT/data_input}" \
    --output-dir "${OUTPUT_DIR:-data_output}" \
    --output-model-dir "${OUTPUT_MODEL_DIR:-fine_tuned_weights}" \
    --epochs "${NUM_EPOCHS:-10}" \
    --batch-size "${BATCH_SIZE:-1}" \
    --questions "${NUM_QUESTIONS:-3}" \
    --augment-paraphrases "${AUGMENT_PARAPHRASES:-0}"

echo "Pipeline job completed at $(date)"


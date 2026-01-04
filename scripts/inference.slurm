#!/bin/bash
#SBATCH --job-name=qa_pipeline
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
mkdir -p logs
cd $SLURM_SUBMIT_DIR

bash scripts/run_pipeline.sh \
    --model "${MODEL_NAME:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}" \
    --epochs "${NUM_EPOCHS:-10}" \
    --batch-size "${BATCH_SIZE:-2}" \
    --questions "${NUM_QUESTIONS:-3}"

echo "Pipeline job completed at $(date)"


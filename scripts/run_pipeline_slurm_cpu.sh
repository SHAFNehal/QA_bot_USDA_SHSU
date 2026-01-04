#!/bin/bash
#SBATCH --job-name=qa_pipeline_cpu
#SBATCH --output=logs/pipeline_cpu_%j.out
#SBATCH --error=logs/pipeline_cpu_%j.err
#SBATCH --time=48:00:00              # CPU daha yavaş, daha uzun timeout
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16           # CPU için daha fazla core
#SBATCH --mem=64G                    # CPU için daha fazla RAM
#SBATCH --partition=cpu              # CPU partition (cluster'ınıza göre değiştirin)

# Aktif dizine geç
cd $SLURM_SUBMIT_DIR

# Log klasörünü oluştur
mkdir -p logs

# CPU modunda çalıştır (device=cpu)
bash scripts/run_pipeline.sh \
    --model "${MODEL_NAME:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}" \
    --epochs "${NUM_EPOCHS:-10}" \
    --batch-size "${BATCH_SIZE:-1}" \
    --questions "${NUM_QUESTIONS:-3}"

echo "Pipeline job completed at $(date)"


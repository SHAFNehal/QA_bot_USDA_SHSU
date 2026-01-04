#!/bin/bash
#SBATCH --job-name=qa_pipeline
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err
#SBATCH --time=24:00:00              # 24 saat timeout (ihtiyaca göre ayarlayın)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8           # CPU sayısı (ihtiyaca göre ayarlayın)
#SBATCH --mem=32G                    # RAM miktarı (ihtiyaca göre ayarlayın)
#SBATCH --gres=gpu:1                 # GPU kullanımı (GPU yoksa bu satırı silin veya yorum yapın)
#SBATCH --partition=gpu              # Partition adı (cluster'ınıza göre değiştirin: gpu, gpu_v100, etc.)

# SLURM ortam değişkenlerini ayarla
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# Log klasörünü oluştur
mkdir -p logs

# Aktif dizine geç
cd $SLURM_SUBMIT_DIR

# Python path'i ayarla (gerekirse)
# export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Pipeline'ı çalıştır
# Tüm parametreleri buraya ekleyebilirsiniz
bash scripts/run_pipeline.sh \
    --model "${MODEL_NAME:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}" \
    --epochs "${NUM_EPOCHS:-10}" \
    --batch-size "${BATCH_SIZE:-2}" \
    --questions "${NUM_QUESTIONS:-3}"

echo "Pipeline job completed at $(date)"


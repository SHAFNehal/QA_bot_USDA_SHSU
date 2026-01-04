#!/bin/bash
#SBATCH --job-name=qa_training
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err
#SBATCH --time=12:00:00              # Sadece training için daha kısa süre
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1                 # GPU kullanımı
#SBATCH --partition=gpu

# Aktif dizine geç
cd $SLURM_SUBMIT_DIR

# Log klasörünü oluştur
mkdir -p logs

# Sadece training adımını çalıştır (dataset hazırsa)
python src/training/fine_tuner.py \
    --dataset_path "${DATASET_PATH:-data_output/training_dataset.jsonl}" \
    --output_dir "${OUTPUT_DIR:-fine_tuned_weights}" \
    --model_name "${MODEL_NAME:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}" \
    --num_train_epochs "${NUM_EPOCHS:-10}" \
    --per_device_train_batch_size "${BATCH_SIZE:-2}" \
    --gradient_accumulation_steps 4 \
    --early_stopping_patience 3

echo "Training job completed at $(date)"


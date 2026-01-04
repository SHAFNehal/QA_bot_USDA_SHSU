#!/bin/bash
#
# SLURM Pipeline Submission Script
# 
# Bu script pipeline'ı SLURM'a submit eder ve job ID'yi gösterir
#
# Kullanım:
#   ./submit_pipeline.sh                    # GPU ile çalıştır
#   ./submit_pipeline.sh --cpu               # CPU ile çalıştır
#   ./submit_pipeline.sh --model microsoft/phi-3-mini-4k-instruct
#

# Varsayılan ayarlar
USE_GPU=true
MODEL_NAME="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
NUM_EPOCHS=10
BATCH_SIZE=2
NUM_QUESTIONS=3

# Parametreleri parse et
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu)
            USE_GPU=false
            shift
            ;;
        --model|-m)
            MODEL_NAME="$2"
            shift 2
            ;;
        --epochs|-e)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --batch-size|-b)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --questions|-q)
            NUM_QUESTIONS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Submit pipeline job to SLURM"
            echo ""
            echo "Options:"
            echo "  --cpu              Use CPU instead of GPU"
            echo "  -m, --model MODEL  Model name"
            echo "  -e, --epochs NUM   Number of epochs"
            echo "  -b, --batch-size   Batch size"
            echo "  -q, --questions    Questions per chunk"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Log klasörünü oluştur
mkdir -p logs

# Environment variables'ı export et
export MODEL_NAME="$MODEL_NAME"
export NUM_EPOCHS="$NUM_EPOCHS"
export BATCH_SIZE="$BATCH_SIZE"
export NUM_QUESTIONS="$NUM_QUESTIONS"

# GPU veya CPU script'ini seç
if [ "$USE_GPU" = true ]; then
    SCRIPT="scripts/run_pipeline_slurm.sh"
    echo "Submitting GPU job..."
else
    SCRIPT="scripts/run_pipeline_slurm_cpu.sh"
    echo "Submitting CPU job..."
fi

# Job'u submit et
JOB_ID=$(sbatch --parsable "$SCRIPT")

if [ $? -eq 0 ]; then
    echo "✓ Job submitted successfully!"
    echo "  Job ID: $JOB_ID"
    echo "  Model: $MODEL_NAME"
    echo "  Epochs: $NUM_EPOCHS"
    echo ""
    echo "Monitor job with:"
    echo "  squeue -j $JOB_ID"
    echo ""
    echo "View output:"
    echo "  tail -f logs/pipeline_${JOB_ID}.out"
    echo ""
    echo "Cancel job:"
    echo "  scancel $JOB_ID"
else
    echo "✗ Job submission failed!"
    exit 1
fi


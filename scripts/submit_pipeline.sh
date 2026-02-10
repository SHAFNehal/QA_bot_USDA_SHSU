#!/bin/bash
#
# SLURM Pipeline Submission Script
# 
# This script submits the pipeline to SLURM and displays the job ID
#
# Usage:
#   ./submit_pipeline.sh                    # Run with GPU
#   ./submit_pipeline.sh --cpu              # Run with CPU
#   ./submit_pipeline.sh --model microsoft/phi-3-mini-4k-instruct
#

# Default settings
USE_GPU=true
MODEL_NAME="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
INPUT_DIR=""
OUTPUT_DIR="data_output"
OUTPUT_MODEL_DIR="fine_tuned_weights"
NUM_EPOCHS=10
BATCH_SIZE=1
NUM_QUESTIONS=3
AUGMENT_PARAPHRASES=0

# Parse parameters
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
        --input-dir|-i)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output-dir|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --output-model-dir)
            OUTPUT_MODEL_DIR="$2"
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
        --augment-paraphrases|-a)
            AUGMENT_PARAPHRASES="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Submit pipeline job to SLURM"
            echo ""
            echo "Options:"
            echo "  --cpu                   Use CPU instead of GPU"
            echo "  -m, --model MODEL       Model name"
            echo "  -i, --input-dir DIR     Input directory for documents"
            echo "  -o, --output-dir DIR    Output directory for datasets"
            echo "  --output-model-dir DIR  Output directory for model"
            echo "  -e, --epochs NUM        Number of epochs"
            echo "  -b, --batch-size NUM    Batch size"
            echo "  -q, --questions NUM     Questions per chunk"
            echo "  -a, --augment-paraphrases NUM  Number of paraphrase variations"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create log directory
mkdir -p logs

# Export environment variables
export MODEL_NAME="$MODEL_NAME"
export NUM_EPOCHS="$NUM_EPOCHS"
export BATCH_SIZE="$BATCH_SIZE"
export NUM_QUESTIONS="$NUM_QUESTIONS"
export OUTPUT_DIR="$OUTPUT_DIR"
export OUTPUT_MODEL_DIR="$OUTPUT_MODEL_DIR"
export AUGMENT_PARAPHRASES="$AUGMENT_PARAPHRASES"

# Set INPUT_DIR only if provided
if [ -n "$INPUT_DIR" ]; then
    export INPUT_DIR="$INPUT_DIR"
fi

# Select GPU or CPU script
if [ "$USE_GPU" = true ]; then
    SCRIPT="scripts/run_pipeline_slurm.sh"
    echo "Submitting GPU job..."
else
    SCRIPT="scripts/run_pipeline_slurm_cpu.sh"
    echo "Submitting CPU job..."
fi

# Submit job
JOB_ID=$(sbatch --parsable "$SCRIPT")

if [ $? -eq 0 ]; then
    echo "Success: Job submitted successfully!"
    echo "  Job ID: $JOB_ID"
    echo "  Model: $MODEL_NAME"
    echo "  Epochs: $NUM_EPOCHS"
    echo "  Batch size: $BATCH_SIZE"
    echo "  Output dir: $OUTPUT_DIR"
    echo "  Model dir: $OUTPUT_MODEL_DIR"
    echo ""
    echo "Monitor job with:"
    echo "  squeue -j $JOB_ID"
    echo ""
    echo "View output:"
    if [ "$USE_GPU" = true ]; then
        echo "  tail -f logs/pipeline_${JOB_ID}.out"
    else
        echo "  tail -f logs/pipeline_cpu_${JOB_ID}.out"
    fi
    echo ""
    echo "Cancel job:"
    echo "  scancel $JOB_ID"
else
    echo "Error: Job submission failed!"
    exit 1
fi
    exit 1
fi


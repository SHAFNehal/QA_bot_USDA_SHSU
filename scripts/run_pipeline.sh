#!/bin/bash
#
# End-to-End Pipeline Script for QA Bot Fine-tuning
#
# This script runs the complete training pipeline:
# 1. Generate QA dataset from input documents
# 2. Clean and validate the dataset
# 3. Generate multi-turn conversation data
# 4. Merge all datasets (QA + conversational + multi-turn)
# 5. Fine-tune the model with LoRA
# 6. Evaluate the model performance
# 7. Save fine-tuned model weights
#
# Usage:
#   ./run_pipeline.sh                                    # Run full pipeline with defaults
#   ./run_pipeline.sh --model microsoft/phi-3-mini-4k-instruct  # Use Phi-3 instead of TinyLlama
#   ./run_pipeline.sh --skip-generation                  # Skip QA generation (use existing dataset)
#   ./run_pipeline.sh --eval-only                        # Only run evaluation
#   ./run_pipeline.sh --model meta-llama/Meta-Llama-3-8B-Instruct --epochs 15  # Custom model and epochs
#

set -e  # Exit on error

# Set PYTHONPATH to project root so Python can find 'src' module
# This ensures imports like 'from src.utils...' work correctly
PROJECT_ROOT="/home/ybd002/llm"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Default Configuration (can be overridden by environment variables or command-line args)
INPUT_DIR="${INPUT_DIR:-data_input}"
OUTPUT_DIR="${OUTPUT_DIR:-data_output}"
MODEL_NAME="${MODEL_NAME:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
OUTPUT_MODEL_DIR="${OUTPUT_MODEL_DIR:-fine_tuned_weights}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_QUESTIONS="${NUM_QUESTIONS:-3}"
AUGMENT_PARAPHRASES="${AUGMENT_PARAPHRASES:-0}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Helper functions
print_step() {
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}========================================${NC}\n"
}

print_warning() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}"
}

# Parse command-line arguments
SKIP_GENERATION=false
EVAL_ONLY=false
SKIP_TRAINING=false

while [[ $# -gt 0 ]]; do
    case $1 in
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
        --skip-generation)
            SKIP_GENERATION=true
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --eval-only)
            EVAL_ONLY=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "End-to-end pipeline for QA bot fine-tuning"
            echo ""
            echo "Options:"
            echo "  -m, --model MODEL              Model to use for generation and fine-tuning"
            echo "                                 (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)"
            echo "                                 Examples:"
            echo "                                   - TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            echo "                                   - microsoft/phi-3-mini-4k-instruct"
            echo "                                   - meta-llama/Meta-Llama-3-8B-Instruct"
            echo ""
            echo "  -i, --input-dir DIR            Input directory for documents (default: data_input)"
            echo "  -o, --output-dir DIR           Output directory for datasets (default: data_output)"
            echo "  --output-model-dir DIR         Output directory for model (default: fine_tuned_weights)"
            echo ""
            echo "  -e, --epochs NUM               Number of training epochs (default: 10)"
            echo "  -b, --batch-size NUM           Training batch size (default: 2)"
            echo "  -q, --questions NUM            Questions per chunk during generation (default: 3)"
            echo "  -a, --augment-paraphrases NUM  Number of paraphrase variations (default: 0)"
            echo ""
            echo "  --skip-generation              Skip QA generation step (use existing dataset)"
            echo "  --skip-training                Skip training step"
            echo "  --eval-only                    Only run evaluation"
            echo "  -h, --help                     Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Run full pipeline with TinyLlama (default)"
            echo "  ./run_pipeline.sh"
            echo ""
            echo "  # Use Phi-3 model with 15 epochs"
            echo "  ./run_pipeline.sh --model microsoft/phi-3-mini-4k-instruct --epochs 15"
            echo ""
            echo "  # Use existing dataset, train with Llama 3"
            echo "  ./run_pipeline.sh --skip-generation --model meta-llama/Meta-Llama-3-8B-Instruct"
            echo ""
            echo "  # Generate dataset with paraphrase augmentation"
            echo "  ./run_pipeline.sh --questions 5 --augment-paraphrases 3"
            echo ""
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Skip to evaluation if requested
if [ "$EVAL_ONLY" = true ]; then
    print_step "Evaluation-Only Mode"
    echo "Model:     $MODEL_NAME"
    echo "Model Dir: $OUTPUT_MODEL_DIR"
    echo ""

    if [ ! -d "$OUTPUT_MODEL_DIR" ]; then
        print_error "Model directory not found: $OUTPUT_MODEL_DIR"
        exit 1
    fi

    print_step "Evaluating model"
    python /home/ybd002/llm/src/inference/evaluate.py \
        --model_path "$OUTPUT_MODEL_DIR" \
        --base_model "$MODEL_NAME" \
        --output "$OUTPUT_DIR/evaluation_report.json" \
        --test_data "$OUTPUT_DIR/training_test.jsonl"

    if [ -f "$OUTPUT_DIR/evaluation_report.json" ]; then
        echo ""
        echo "✓ Evaluation completed: $OUTPUT_DIR/evaluation_report.json"
    fi

    print_step "✓ Pipeline completed (evaluation only)!"
    exit 0
fi

# Display configuration
print_step "Pipeline Configuration"
echo "Model:               $MODEL_NAME"
echo "Input Directory:     $INPUT_DIR"
echo "Output Directory:    $OUTPUT_DIR"
echo "Output Model Dir:    $OUTPUT_MODEL_DIR"
echo "Training Epochs:     $NUM_EPOCHS"
echo "Batch Size:          $BATCH_SIZE"
echo "Questions per Chunk: $NUM_QUESTIONS"
if [ "$AUGMENT_PARAPHRASES" -gt 0 ]; then
    echo "Paraphrase Augment:  $AUGMENT_PARAPHRASES variations"
fi
echo ""

# Step 1: Generate QA dataset
if [ "$SKIP_GENERATION" = false ]; then
    print_step "Step 1/7: Generating QA dataset from documents"

    if [ ! -d "$INPUT_DIR" ]; then
        print_error "Input directory not found: $INPUT_DIR"
        print_warning "Please create the directory and add your text files (.txt, .md, .docx)"
        exit 1
    fi

    # Check if input directory has files
    FILE_COUNT=$(find "$INPUT_DIR" -type f \( -name "*.txt" -o -name "*.md" -o -name "*.docx" \) | wc -l)
    if [ "$FILE_COUNT" -eq 0 ]; then
        print_error "No supported files found in $INPUT_DIR"
        print_warning "Please add .txt, .md, or .docx files"
        exit 1
    fi

    echo "Found $FILE_COUNT document(s) to process"

    # Build the command with optional paraphrase augmentation
    CMD="python /home/ybd002/llm/src/dataset/dataset_creator.py \
        --input_dir \"$INPUT_DIR\" \
        --output_file \"$OUTPUT_DIR/qa_dataset.jsonl\" \
        --model_name \"$MODEL_NAME\" \
        --num_questions $NUM_QUESTIONS"

    if [ "$AUGMENT_PARAPHRASES" -gt 0 ]; then
        CMD="$CMD --augment_paraphrases $AUGMENT_PARAPHRASES"
    fi

    eval $CMD

    echo "QA dataset generated: $OUTPUT_DIR/qa_dataset.jsonl"
else
    print_warning "Skipping QA generation step"
    if [ ! -f "$OUTPUT_DIR/qa_dataset.jsonl" ]; then
        print_error "No existing QA dataset found at $OUTPUT_DIR/qa_dataset.jsonl"
        exit 1
    fi
fi

# Step 2: Clean and validate the dataset
print_step "Step 2/7: Cleaning and validating dataset"

python /home/ybd002/llm/src/dataset/data_cleaner.py \
    --input "$OUTPUT_DIR/qa_dataset.jsonl" \
    --output "$OUTPUT_DIR/qa_cleaned.jsonl" \
    --relaxed

QA_COUNT=$(wc -l < "$OUTPUT_DIR/qa_cleaned.jsonl")
echo "Cleaned dataset: $OUTPUT_DIR/qa_cleaned.jsonl ($QA_COUNT pairs)"

# Step 3: Generate multi-turn conversation data
print_step "Step 3/7: Generating multi-turn conversation data"

python /home/ybd002/llm/src/dataset/generators/multiturn_generator.py \
    --input "$OUTPUT_DIR/qa_cleaned.jsonl" \
    --output "$OUTPUT_DIR/multiturn.jsonl" \
    --include_coreference

MULTITURN_COUNT=$(wc -l < "$OUTPUT_DIR/multiturn.jsonl")
echo "Multi-turn data: $OUTPUT_DIR/multiturn.jsonl ($MULTITURN_COUNT examples)"

# Step 4: Merge all datasets
print_step "Step 4/7: Merging datasets (QA + conversational + multi-turn)"

python /home/ybd002/llm/src/dataset/merge_datasets.py \
    --qa_data "$OUTPUT_DIR/qa_cleaned.jsonl" \
    --multiturn_data "$OUTPUT_DIR/multiturn.jsonl" \
    --output "$OUTPUT_DIR/training_dataset.jsonl" \
    --conversational_multiplier 3

# Count final dataset size
DATASET_SIZE=$(wc -l < "$OUTPUT_DIR/training_dataset.jsonl")
echo "Final training dataset: $OUTPUT_DIR/training_dataset.jsonl"
echo "Total training examples: $DATASET_SIZE"

# Step 5: Fine-tune the model
if [ "$SKIP_TRAINING" = false ]; then
    print_step "Step 5/7: Fine-tuning the model with LoRA"

    echo "Training with:"
    echo "  - Model: $MODEL_NAME"
    echo "  - Epochs: $NUM_EPOCHS"
    echo "  - Batch size: $BATCH_SIZE"
    echo "  - Dataset size: $DATASET_SIZE examples"
    echo ""

    python /home/ybd002/llm/src/training/fine_tuner.py \
        --dataset_path "$OUTPUT_DIR/training_dataset.jsonl" \
        --output_dir "$OUTPUT_MODEL_DIR" \
        --model_name "$MODEL_NAME" \
        --num_train_epochs "$NUM_EPOCHS" \
        --per_device_train_batch_size "$BATCH_SIZE" \
        --gradient_accumulation_steps 8 \
        --early_stopping_patience 3 \
        --holdout_output_path "$OUTPUT_DIR/training_test.jsonl"

    if [ -d "$OUTPUT_MODEL_DIR" ]; then
        echo "✓ Fine-tuning completed successfully!"
    else
        print_error "Fine-tuning failed - model directory not created"
        exit 1
    fi
else
    print_warning "Skipping training step"
fi

# Step 6: Save model weights
if [ -d "$OUTPUT_MODEL_DIR" ]; then
    print_step "Step 6/7: Saving fine-tuned model weights"

    # Check model files
    if [ -f "$OUTPUT_MODEL_DIR/adapter_config.json" ] && [ -f "$OUTPUT_MODEL_DIR/adapter_model.safetensors" ]; then
        MODEL_SIZE=$(du -sh "$OUTPUT_MODEL_DIR" | cut -f1)
        echo "✓ LoRA adapter weights saved: $OUTPUT_MODEL_DIR"
        echo "✓ Model size: $MODEL_SIZE"
        echo ""
        echo "Model components:"
        ls -lh "$OUTPUT_MODEL_DIR" | grep -E "adapter_|config|tokenizer" || true
    else
        print_warning "Model files may be incomplete"
    fi
else
    print_warning "Model directory not found, skipping save verification"
fi

# Step 7: Evaluate the model
print_step "Step 7/7: Evaluating the model performance"

if [ -d "$OUTPUT_MODEL_DIR" ]; then
    python /home/ybd002/llm/src/inference/evaluate.py \
        --model_path "$OUTPUT_MODEL_DIR" \
        --base_model "$MODEL_NAME" \
        --output "$OUTPUT_DIR/evaluation_report.json" \
        --test_data "$OUTPUT_DIR/training_test.jsonl"

    if [ -f "$OUTPUT_DIR/evaluation_report.json" ]; then
        echo "✓ Evaluation completed: $OUTPUT_DIR/evaluation_report.json"
    fi
else
    print_warning "Model directory not found, skipping evaluation"
fi

# Summary
print_step "✓ Pipeline completed successfully!"
echo ""
echo "=================================="
echo "Pipeline Summary"
echo "=================================="
echo ""
echo "Configuration:"
echo "  Model:             $MODEL_NAME"
echo "  Training Epochs:   $NUM_EPOCHS"
echo "  Batch Size:        $BATCH_SIZE"
echo ""
echo "Output Files:"
echo "  1. QA Dataset:        $OUTPUT_DIR/qa_dataset.jsonl"
echo "  2. Cleaned Dataset:   $OUTPUT_DIR/qa_cleaned.jsonl ($QA_COUNT pairs)"
echo "  3. Multi-turn Data:   $OUTPUT_DIR/multiturn.jsonl ($MULTITURN_COUNT examples)"
echo "  4. Training Dataset:  $OUTPUT_DIR/training_dataset.jsonl ($DATASET_SIZE examples)"
if [ -f "$OUTPUT_DIR/training_test.jsonl" ]; then
    echo "  5. Holdout Test Set:   $OUTPUT_DIR/training_test.jsonl (for evaluation)"
fi
if [ -d "$OUTPUT_MODEL_DIR" ]; then
    echo "  6. Fine-tuned Model:  $OUTPUT_MODEL_DIR/ ($(du -sh "$OUTPUT_MODEL_DIR" | cut -f1))"
fi
if [ -f "$OUTPUT_DIR/evaluation_report.json" ]; then
    echo "  7. Evaluation Report: $OUTPUT_DIR/evaluation_report.json"
fi
echo ""
echo "=================================="
echo "Next Steps:"
echo "=================================="
echo ""
echo "1. Test the model interactively:"
echo "   python /home/ybd002/llm/src/inference/inference.py --peft_model $OUTPUT_MODEL_DIR --interactive"
echo ""
echo "2. Run batch evaluation (with holdout test set):"
echo "   python /home/ybd002/llm/src/inference/evaluate.py --model_path $OUTPUT_MODEL_DIR --base_model \"$MODEL_NAME\" --test_data $OUTPUT_DIR/training_test.jsonl"
echo ""
echo "3. View evaluation results:"
echo "   cat $OUTPUT_DIR/evaluation_report.json"
echo ""

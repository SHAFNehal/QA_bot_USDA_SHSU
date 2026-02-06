#!/bin/bash
#
# Evaluate an existing trained model on newly generated QA data from a document folder.
#
# Use case: You have trained weights and want to evaluate the model on a set of
# documents by (1) generating question-answer pairs from those documents, then
# (2) running the trained model on the questions and comparing to the generated
# answers. The evaluation report is saved in the output directory (no preset
# greeting/gratitude tests—only the generated data).
#
# Steps:
# 1. Read document folder (like training pipeline input)
# 2. Generate QA pairs from documents using the base model
# 3. Evaluate the trained model (PEFT weights) on the generated QA
# 4. Save evaluation report to output directory
#
# Usage:
#   ./run_eval_on_documents.sh --peft-model fine_tuned_weights --input-dir data_input --output-dir eval_output
#   ./run_eval_on_documents.sh --peft-model ./weights --input-dir ./my_docs -o ./eval_out -q 5
#

set -e

# Project root: directory containing 'src' (one level up from scripts/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Defaults
INPUT_DIR="${INPUT_DIR:-data_input}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_output}"
MODEL_NAME="${MODEL_NAME:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
NUM_QUESTIONS="${NUM_QUESTIONS:-3}"
PEFT_MODEL=""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_step() {
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}========================================${NC}\n"
}

print_warning() { echo -e "${YELLOW}WARNING: $1${NC}"; }
print_error() { echo -e "${RED}ERROR: $1${NC}"; }

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --peft-model)
            PEFT_MODEL="$2"
            shift 2
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
        --questions|-q)
            NUM_QUESTIONS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Evaluate an existing trained model on QA generated from a document folder."
            echo ""
            echo "Required:"
            echo "  --peft-model PATH    Path to trained model weights (LoRA adapter or full model)"
            echo ""
            echo "Options:"
            echo "  -m, --model NAME     Base model name for QA generation (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)"
            echo "  -i, --input-dir DIR  Folder with .txt, .md, .docx documents (default: data_input)"
            echo "  -o, --output-dir DIR Where to save generated QA and evaluation report (default: eval_output)"
            echo "  -q, --questions N    Questions per chunk when generating QA (default: 3)"
            echo "  -h, --help           Show this help"
            echo ""
            echo "Output files (in OUTPUT_DIR):"
            echo "  generated_qa_eval.jsonl  - Generated question-answer pairs"
            echo "  evaluation_report.json   - Evaluation results (same format as pipeline)"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$PEFT_MODEL" ]; then
    print_error "Required: --peft-model PATH (path to trained weights)"
    exit 1
fi

if [ ! -d "$PEFT_MODEL" ]; then
    print_error "Model directory not found: $PEFT_MODEL"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Config summary
print_step "Evaluate-on-documents configuration"
echo "Trained model (PEFT): $PEFT_MODEL"
echo "Base model (for QA gen): $MODEL_NAME"
echo "Input directory:      $INPUT_DIR"
echo "Output directory:     $OUTPUT_DIR"
echo "Questions per chunk:  $NUM_QUESTIONS"
echo ""

# Step 1: Generate QA from document folder
print_step "Step 1/2: Generating QA from documents"

if [ ! -d "$INPUT_DIR" ]; then
    print_error "Input directory not found: $INPUT_DIR"
    exit 1
fi

FILE_COUNT=$(find "$INPUT_DIR" -type f \( -name "*.txt" -o -name "*.md" -o -name "*.docx" \) 2>/dev/null | wc -l)
if [ "$FILE_COUNT" -eq 0 ]; then
    print_error "No .txt, .md, or .docx files in $INPUT_DIR"
    exit 1
fi

echo "Found $FILE_COUNT document(s). Generating QA..."

python "$PROJECT_ROOT/src/dataset/dataset_creator.py" \
    --input_dir "$INPUT_DIR" \
    --output_file "$OUTPUT_DIR/generated_qa_eval.jsonl" \
    --model_name "$MODEL_NAME" \
    --num_questions "$NUM_QUESTIONS"

if [ ! -f "$OUTPUT_DIR/generated_qa_eval.jsonl" ]; then
    print_error "QA generation did not produce $OUTPUT_DIR/generated_qa_eval.jsonl"
    exit 1
fi

QA_COUNT=$(wc -l < "$OUTPUT_DIR/generated_qa_eval.jsonl")
echo "Generated $QA_COUNT QA pairs: $OUTPUT_DIR/generated_qa_eval.jsonl"

# Step 2: Evaluate trained model on generated data only
print_step "Step 2/2: Evaluating model on generated data"

python "$PROJECT_ROOT/src/inference/evaluate.py" \
    --model_path "$PEFT_MODEL" \
    --base_model "$MODEL_NAME" \
    --test_data "$OUTPUT_DIR/generated_qa_eval.jsonl" \
    --test_data_only \
    --output "$OUTPUT_DIR/evaluation_report.json"

print_step "Done"
echo "Output files:"
echo "  Generated QA:    $OUTPUT_DIR/generated_qa_eval.jsonl ($QA_COUNT pairs)"
echo "  Evaluation report: $OUTPUT_DIR/evaluation_report.json"
echo ""

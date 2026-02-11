#!/bin/bash

################################################################################
# Extended Pipeline with RAG and Hybrid RAG
# 
# This script runs the complete pipeline including:
# 1. Dataset generation from documents
# 2. Fine-tuning
# 3. Fine-tuned model evaluation
# 4. RAG database creation
# 5. RAG evaluation
# 6. Hybrid RAG evaluation
# 7. Comparison report
#
# Usage:
#   ./scripts/run_full_pipeline_with_rag.sh
#   ./scripts/run_full_pipeline_with_rag.sh --model_name "microsoft/Phi-3-mini-4k-instruct"
################################################################################

set -e  # Exit on error

# Get the absolute path to the project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

# Default parameters
MODEL_NAME="${MODEL_NAME:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
INPUT_DIR="${INPUT_DIR:-data_input}"
OUTPUT_DIR="${OUTPUT_DIR:-data_output}"
OUTPUT_MODEL_DIR="${OUTPUT_MODEL_DIR:-fine_tuned_weights}"
RAG_DB_PATH="${RAG_DB_PATH:-rag_db}"
HYBRID_RAG_DB_PATH="${HYBRID_RAG_DB_PATH:-rag_db_hybrid}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --input_dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --output_model_dir)
            OUTPUT_MODEL_DIR="$2"
            shift 2
            ;;
        --rag_db_path)
            RAG_DB_PATH="$2"
            shift 2
            ;;
        --hybrid_rag_db_path)
            HYBRID_RAG_DB_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Helper function for printing steps
print_step() {
    echo ""
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

# Create necessary directories
print_step "Setting up directories"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_MODEL_DIR"
mkdir -p logs

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Input: $INPUT_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Fine-tuned weights: $OUTPUT_MODEL_DIR"
echo "  RAG DB: $RAG_DB_PATH"
echo "  Hybrid RAG DB: $HYBRID_RAG_DB_PATH"

################################################################################
# PHASE 1: Fine-tuning Pipeline
################################################################################

print_step "PHASE 1: Fine-tuning Pipeline"

# Run the main pipeline (generation + training + evaluation)
echo "Running main pipeline..."
"$PROJECT_ROOT/scripts/run_pipeline.sh" \
    --model_name "$MODEL_NAME" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --output_model_dir "$OUTPUT_MODEL_DIR"

if [ ! -f "$OUTPUT_DIR/evaluation_report.json" ]; then
    echo "Error: Fine-tuned evaluation report not found"
    exit 1
fi

echo "✓ Phase 1 complete: Fine-tuned model ready"

################################################################################
# PHASE 2: RAG Setup
################################################################################

print_step "PHASE 2: RAG Setup"

# Ingest documents for RAG
echo "Ingesting documents for RAG..."
python "$PROJECT_ROOT/src/rag/ingest.py" \
    --input_dir "$INPUT_DIR" \
    --db_path "$RAG_DB_PATH"

if [ ! -d "$RAG_DB_PATH" ]; then
    echo "Error: RAG database not created"
    exit 1
fi

echo "✓ Phase 2 complete: RAG database ready"

################################################################################
# PHASE 3: RAG Evaluation
################################################################################

print_step "PHASE 3: RAG Evaluation"

# Evaluate RAG system
echo "Evaluating RAG system..."
python "$PROJECT_ROOT/src/rag/evaluate_rag.py" \
    --test_data "$OUTPUT_DIR/training_test.jsonl" \
    --db_path "$RAG_DB_PATH" \
    --base_model "$MODEL_NAME" \
    --output "$OUTPUT_DIR/rag_evaluation_report.json"

if [ -f "$OUTPUT_DIR/rag_evaluation_report.json" ]; then
    echo "✓ Phase 3 complete: RAG evaluation done"
else
    echo "Warning: RAG evaluation report not found"
fi

################################################################################
# PHASE 4: Hybrid RAG Setup
################################################################################

print_step "PHASE 4: Hybrid RAG Setup"

# Ingest documents for Hybrid RAG (separate database)
echo "Ingesting documents for Hybrid RAG..."
python "$PROJECT_ROOT/src/Hybrid_RAG/ingest_for_hybrid.py" \
    --input_dir "$INPUT_DIR" \
    --db_path "$HYBRID_RAG_DB_PATH"

if [ ! -d "$HYBRID_RAG_DB_PATH" ]; then
    echo "Error: Hybrid RAG database not created"
    exit 1
fi

echo "✓ Phase 4 complete: Hybrid RAG database ready"

################################################################################
# PHASE 5: Hybrid RAG Evaluation
################################################################################

print_step "PHASE 5: Hybrid RAG Evaluation"

# Evaluate Hybrid RAG system
echo "Evaluating Hybrid RAG system..."
python "$PROJECT_ROOT/src/Hybrid_RAG/evaluate_hybrid_rag.py" \
    --test_data "$OUTPUT_DIR/training_test.jsonl" \
    --model_path "$OUTPUT_MODEL_DIR" \
    --db_path "$HYBRID_RAG_DB_PATH" \
    --base_model "$MODEL_NAME" \
    --output "$OUTPUT_DIR/hybrid_rag_evaluation_report.json"

if [ -f "$OUTPUT_DIR/hybrid_rag_evaluation_report.json" ]; then
    echo "✓ Phase 5 complete: Hybrid RAG evaluation done"
else
    echo "Warning: Hybrid RAG evaluation report not found"
fi

################################################################################
# PHASE 6: Comparison Report
################################################################################

print_step "PHASE 6: Generating Comparison Report"

# Create comparison report
python << 'EOF'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])

# Load all reports
reports = {}
for report_name, filename in [
    ("Fine-tuned", "evaluation_report.json"),
    ("RAG", "rag_evaluation_report.json"),
    ("Hybrid RAG", "hybrid_rag_evaluation_report.json"),
]:
    path = output_dir / filename
    if path.exists():
        with open(path) as f:
            reports[report_name] = json.load(f)
    else:
        print(f"Warning: {filename} not found", file=sys.stderr)

# Generate comparison
comparison = {
    "timestamp": reports.get("Fine-tuned", {}).get("timestamp", ""),
    "systems": {}
}

for system_name, report in reports.items():
    comparison["systems"][system_name] = {
        "overall_score": report.get("overall_score", "N/A"),
        "passed_tests": report.get("passed_tests", 0),
        "total_tests": report.get("total_tests", 0),
        "avg_response_time": report.get("avg_response_time", "N/A"),
    }
    
    # Add answer metrics if available
    answer_metrics = report.get("answer_metrics", {}) or report.get("standard_metrics_agg", {})
    if answer_metrics:
        comparison["systems"][system_name]["answer_metrics"] = answer_metrics

# Save comparison
comparison_path = output_dir / "comparison_report.json"
with open(comparison_path, 'w') as f:
    json.dump(comparison, f, indent=2)

# Print comparison
print("\n" + "=" * 70)
print("COMPARISON REPORT")
print("=" * 70)

for system_name in ["Fine-tuned", "RAG", "Hybrid RAG"]:
    if system_name in comparison["systems"]:
        data = comparison["systems"][system_name]
        print(f"\n{system_name}:")
        print(f"  Overall Score: {data['overall_score']}")
        print(f"  Tests Passed: {data['passed_tests']}/{data['total_tests']}")
        print(f"  Avg Response Time: {data['avg_response_time']}")
        
        if "answer_metrics" in data:
            print(f"  Answer Metrics:")
            for metric, value in data["answer_metrics"].items():
                print(f"    {metric}: {value:.4f}")

print("\n" + "=" * 70)
print(f"✓ Comparison report saved: {comparison_path}")
print("=" * 70)
EOF
python -c "$(cat)" "$OUTPUT_DIR"

################################################################################
# Summary
################################################################################

print_step "PIPELINE COMPLETE"

echo ""
echo "All phases completed successfully!"
echo ""
echo "Output files:"
echo "  1. Fine-tuned model: $OUTPUT_MODEL_DIR/"
echo "  2. Fine-tuned evaluation: $OUTPUT_DIR/evaluation_report.json"
echo "  3. RAG database: $RAG_DB_PATH/"
echo "  4. RAG evaluation: $OUTPUT_DIR/rag_evaluation_report.json"
echo "  5. Hybrid RAG database: $HYBRID_RAG_DB_PATH/"
echo "  6. Hybrid RAG evaluation: $OUTPUT_DIR/hybrid_rag_evaluation_report.json"
echo "  7. Comparison report: $OUTPUT_DIR/comparison_report.json"
echo ""
echo "Next steps:"
echo "  1. Review comparison report: cat $OUTPUT_DIR/comparison_report.json | python -m json.tool"
echo "  2. Test fine-tuned model: python src/inference/inference.py --peft-model $OUTPUT_MODEL_DIR"
echo "  3. Test RAG: python src/rag/query.py"
echo "  4. Test Hybrid RAG: python src/Hybrid_RAG/query.py --peft_model_path $OUTPUT_MODEL_DIR"
echo ""

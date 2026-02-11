#!/bin/bash

################################################################################
# Comparison Benchmark Script
# 
# Runs evaluation on all three systems and generates comparison report:
# - Fine-tuned model
# - RAG
# - Hybrid RAG
#
# Prerequisites:
# - Fine-tuned model must exist
# - RAG databases must be set up
# - Test data must exist
#
# Usage:
#   ./scripts/run_comparison.sh
#   ./scripts/run_comparison.sh --test_data data_output/training_test.jsonl
################################################################################

set -e  # Exit on error

# Get the absolute path to the project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

# Default parameters
MODEL_NAME="${MODEL_NAME:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
MODEL_PATH="${MODEL_PATH:-fine_tuned_weights}"
TEST_DATA="${TEST_DATA:-data_output/training_test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-data_output}"
RAG_DB_PATH="${RAG_DB_PATH:-rag_db}"
HYBRID_RAG_DB_PATH="${HYBRID_RAG_DB_PATH:-rag_db_hybrid}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --test_data)
            TEST_DATA="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --rag_db)
            RAG_DB_PATH="$2"
            shift 2
            ;;
        --hybrid_rag_db)
            HYBRID_RAG_DB_PATH="$2"
            shift 2
            ;;
        --base_model)
            MODEL_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--model_path PATH] [--test_data PATH] [--output_dir PATH]"
            exit 1
            ;;
    esac
done

# Helper function
print_step() {
    echo ""
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

print_step "COMPARISON BENCHMARK"

echo "Configuration:"
echo "  Base Model: $MODEL_NAME"
echo "  Fine-tuned Model: $MODEL_PATH"
echo "  Test Data: $TEST_DATA"
echo "  Output: $OUTPUT_DIR"
echo "  RAG DB: $RAG_DB_PATH"
echo "  Hybrid RAG DB: $HYBRID_RAG_DB_PATH"

# Verify prerequisites
if [ ! -f "$TEST_DATA" ]; then
    echo "Error: Test data not found: $TEST_DATA"
    exit 1
fi

MISSING=""
if [ ! -d "$MODEL_PATH" ]; then
    echo "Warning: Fine-tuned model not found: $MODEL_PATH"
    MISSING="$MISSING fine-tuned"
fi

if [ ! -d "$RAG_DB_PATH" ]; then
    echo "Warning: RAG database not found: $RAG_DB_PATH"
    MISSING="$MISSING RAG"
fi

if [ ! -d "$HYBRID_RAG_DB_PATH" ]; then
    echo "Warning: Hybrid RAG database not found: $HYBRID_RAG_DB_PATH"
    MISSING="$MISSING hybrid"
fi

if [ -n "$MISSING" ]; then
    echo ""
    echo "Missing components:$MISSING"
    echo ""
    echo "To set up all components, run:"
    echo "  ./scripts/run_full_pipeline_with_rag.sh"
    echo ""
    exit 1
fi

################################################################################
# Run Evaluations
################################################################################

print_step "1/3: Evaluating Fine-tuned Model"

python "$PROJECT_ROOT/src/inference/evaluate.py" \
    --model_path "$MODEL_PATH" \
    --base_model "$MODEL_NAME" \
    --test_data "$TEST_DATA" \
    --output "$OUTPUT_DIR/evaluation_report.json"

echo "✓ Fine-tuned evaluation complete"

print_step "2/3: Evaluating RAG System"

python "$PROJECT_ROOT/src/rag/evaluate_rag.py" \
    --test_data "$TEST_DATA" \
    --db_path "$RAG_DB_PATH" \
    --base_model "$MODEL_NAME" \
    --output "$OUTPUT_DIR/rag_evaluation_report.json"

echo "✓ RAG evaluation complete"

print_step "3/3: Evaluating Hybrid RAG System"

python "$PROJECT_ROOT/src/Hybrid_RAG/evaluate_hybrid_rag.py" \
    --test_data "$TEST_DATA" \
    --model_path "$MODEL_PATH" \
    --db_path "$HYBRID_RAG_DB_PATH" \
    --base_model "$MODEL_NAME" \
    --output "$OUTPUT_DIR/hybrid_rag_evaluation_report.json"

echo "✓ Hybrid RAG evaluation complete"

################################################################################
# Generate Comparison Report
################################################################################

print_step "Generating Comparison Report"

python << 'PYTHON_SCRIPT'
import json
import sys
from pathlib import Path
from datetime import datetime

output_dir = Path(sys.argv[1])

# Load all reports
reports = {}
report_files = {
    "Fine-tuned": "evaluation_report.json",
    "RAG": "rag_evaluation_report.json",
    "Hybrid RAG": "hybrid_rag_evaluation_report.json",
}

for system_name, filename in report_files.items():
    path = output_dir / filename
    if path.exists():
        with open(path) as f:
            reports[system_name] = json.load(f)
    else:
        print(f"Warning: {filename} not found", file=sys.stderr)

if not reports:
    print("Error: No evaluation reports found", file=sys.stderr)
    sys.exit(1)

# Generate detailed comparison
comparison = {
    "timestamp": datetime.now().isoformat(),
    "summary": {},
    "detailed_metrics": {},
    "winner": {}
}

# Compare overall scores
for system_name, report in reports.items():
    overall_score_str = report.get("overall_score", "0%")
    overall_score = float(overall_score_str.replace("%", ""))
    
    comparison["summary"][system_name] = {
        "overall_score": overall_score_str,
        "passed_tests": report.get("passed_tests", 0),
        "total_tests": report.get("total_tests", 0),
        "pass_rate": overall_score,
    }
    
    # Add response time if available
    if "avg_response_time" in report:
        rt_str = report.get("avg_response_time", "0s")
        rt_value = float(rt_str.replace("s", ""))
        comparison["summary"][system_name]["avg_response_time"] = rt_str
        comparison["summary"][system_name]["response_time_value"] = rt_value

# Determine winners
if reports:
    # Best accuracy
    best_accuracy = max(comparison["summary"].items(), key=lambda x: x[1]["pass_rate"])
    comparison["winner"]["best_accuracy"] = {
        "system": best_accuracy[0],
        "score": best_accuracy[1]["overall_score"]
    }
    
    # Fastest (if available)
    systems_with_time = [(name, data["response_time_value"]) 
                         for name, data in comparison["summary"].items() 
                         if "response_time_value" in data]
    if systems_with_time:
        fastest = min(systems_with_time, key=lambda x: x[1])
        comparison["winner"]["fastest"] = {
            "system": fastest[0],
            "time": f"{fastest[1]:.2f}s"
        }

# Collect detailed metrics
for system_name, report in reports.items():
    metrics = report.get("answer_metrics", {}) or report.get("standard_metrics_agg", {})
    if metrics:
        comparison["detailed_metrics"][system_name] = metrics

# Save comparison
comparison_path = output_dir / "comparison_report.json"
with open(comparison_path, 'w') as f:
    json.dump(comparison, f, indent=2)

# Print beautiful comparison table
print("\n" + "=" * 80)
print("SYSTEM COMPARISON REPORT")
print("=" * 80)

# Summary table
print("\n┌─────────────────┬──────────────┬────────────┬───────────────────┐")
print("│     System      │ Overall Score│   Tests    │  Avg Response Time│")
print("├─────────────────┼──────────────┼────────────┼───────────────────┤")

for system_name in ["Fine-tuned", "RAG", "Hybrid RAG"]:
    if system_name in comparison["summary"]:
        data = comparison["summary"][system_name]
        score = data["overall_score"]
        tests = f"{data['passed_tests']}/{data['total_tests']}"
        time = data.get("avg_response_time", "N/A")
        print(f"│ {system_name:15} │ {score:>12} │ {tests:>10} │ {time:>17} │")

print("└─────────────────┴──────────────┴────────────┴───────────────────┘")

# Winners
if "winner" in comparison:
    print("\n🏆 WINNERS:")
    if "best_accuracy" in comparison["winner"]:
        winner = comparison["winner"]["best_accuracy"]
        print(f"  Best Accuracy: {winner['system']} ({winner['score']})")
    if "fastest" in comparison["winner"]:
        winner = comparison["winner"]["fastest"]
        print(f"  Fastest: {winner['system']} ({winner['time']})")

# Detailed metrics comparison
if comparison["detailed_metrics"]:
    print("\n📊 DETAILED METRICS:")
    
    # Get all metric names
    all_metrics = set()
    for metrics in comparison["detailed_metrics"].values():
        all_metrics.update(metrics.keys())
    
    for metric in sorted(all_metrics):
        print(f"\n  {metric}:")
        for system_name in ["Fine-tuned", "RAG", "Hybrid RAG"]:
            if system_name in comparison["detailed_metrics"]:
                value = comparison["detailed_metrics"][system_name].get(metric, "N/A")
                if isinstance(value, float):
                    print(f"    {system_name:15}: {value:.4f}")
                else:
                    print(f"    {system_name:15}: {value}")

print("\n" + "=" * 80)
print(f"✓ Comparison report saved: {comparison_path}")
print("=" * 80)
print()

# Recommendations
print("💡 RECOMMENDATIONS:")
print()

best = comparison["winner"].get("best_accuracy", {}).get("system", "")
fastest = comparison["winner"].get("fastest", {}).get("system", "")

if best == "Fine-tuned":
    print("  • Fine-tuned model has best accuracy - Use for production")
    print("  • Consider this for latency-sensitive applications")
elif best == "RAG":
    print("  • RAG has best accuracy - Good for knowledge-based queries")
    print("  • Easy to update knowledge without retraining")
elif best == "Hybrid RAG":
    print("  • Hybrid RAG has best accuracy - Best of both worlds!")
    print("  • Consider latency trade-off (slower)")

if fastest != best:
    print(f"  • {fastest} is fastest - Consider for real-time applications")

print()
PYTHON_SCRIPT
python3 -c "$(cat)" "$OUTPUT_DIR"

################################################################################
# Done
################################################################################

print_step "COMPARISON COMPLETE"

echo ""
echo "Output files:"
echo "  • Fine-tuned: $OUTPUT_DIR/evaluation_report.json"
echo "  • RAG: $OUTPUT_DIR/rag_evaluation_report.json"
echo "  • Hybrid RAG: $OUTPUT_DIR/hybrid_rag_evaluation_report.json"
echo "  • Comparison: $OUTPUT_DIR/comparison_report.json"
echo ""
echo "View comparison:"
echo "  cat $OUTPUT_DIR/comparison_report.json | python -m json.tool"
echo ""

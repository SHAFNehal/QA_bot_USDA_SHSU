# SLURM Scripts - Quick Reference Guide

## All Scripts Overview

| Script | Purpose | Use When | Key Parameters |
|--------|---------|----------|----------------|
| `run_pipeline.sh` | Full pipeline (local) | Direct bash execution | -m, -i, -o, -e, -b, -q, -a |
| `run_eval_on_documents.sh` | Evaluate on docs | Test model on new docs | --peft-model, -i, -o, -q |
| `run.slurm` | Full pipeline (SLURM) | Submit full job to cluster | All via env vars |
| `run_pipeline_slurm.sh` | Full pipeline GPU | Same as run.slurm | All via env vars |
| `run_pipeline_slurm_cpu.sh` | Full pipeline CPU | No GPU available | All via env vars |
| `run_training_only_slurm.sh` | Training only | Dataset exists, skip gen | MODEL_NAME, NUM_EPOCHS, etc. |
| `inference.slurm` | Inference only | Run chatbot/inference | BASE_MODEL, PEFT_MODEL |
| `submit_pipeline.sh` | Job submitter | Easy SLURM submission | --cpu, -m, -e, -b, -q, -a |

---

## Quick Usage Examples

### 1. Run Full Pipeline Locally
```bash
cd /path/to/QA_bot_USDA_SHSU
./scripts/run_pipeline.sh
```

### 2. Run Pipeline on SLURM (Easiest)
```bash
# GPU mode (default)
./scripts/submit_pipeline.sh

# CPU mode
./scripts/submit_pipeline.sh --cpu

# Custom model and epochs
./scripts/submit_pipeline.sh -m microsoft/phi-3-mini-4k-instruct -e 15
```

### 3. Direct SLURM Submission
```bash
# Set environment variables
export MODEL_NAME="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
export NUM_EPOCHS=10
export BATCH_SIZE=1

# Submit
sbatch scripts/run_pipeline_slurm.sh
```

### 4. Training Only (Dataset Exists)
```bash
export OUTPUT_DIR="data_output"
export OUTPUT_MODEL_DIR="my_fine_tuned_model"
sbatch scripts/run_training_only_slurm.sh
```

### 5. Inference Only
```bash
export BASE_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
export PEFT_MODEL="fine_tuned_weights"
sbatch scripts/inference.slurm
```

### 6. Evaluate Existing Model on New Documents
```bash
./scripts/run_eval_on_documents.sh \
  --peft-model ./fine_tuned_weights \
  --input-dir ./new_documents \
  --output-dir ./eval_results
```

---

## Environment Variables Reference

### Common Variables (All Scripts)
```bash
MODEL_NAME="TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Model to use
NUM_EPOCHS=10                                     # Training epochs
BATCH_SIZE=1                                      # Batch size
NUM_QUESTIONS=3                                   # Questions per chunk
```

### Directory Variables
```bash
INPUT_DIR="data_input"                           # Input documents
OUTPUT_DIR="data_output"                         # Output datasets
OUTPUT_MODEL_DIR="fine_tuned_weights"            # Model output
```

### Advanced Variables
```bash
AUGMENT_PARAPHRASES=0                            # Paraphrase variations (0=off)
GPU_ID=0                                         # Force specific GPU
PROJECT_ROOT=/path/to/project                    # Override project root
```

### Inference-Specific
```bash
BASE_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Base model
PEFT_MODEL="fine_tuned_weights"                  # Fine-tuned weights path
QUESTION="What is machine learning?"             # Optional question
```

---

## Command-Line Arguments (run_pipeline.sh)

```bash
./scripts/run_pipeline.sh [OPTIONS]

Options:
  -m, --model MODEL              Model name
  -i, --input-dir DIR            Input directory
  -o, --output-dir DIR           Output directory
  --output-model-dir DIR         Model output directory
  -e, --epochs NUM               Training epochs
  -b, --batch-size NUM           Batch size
  -q, --questions NUM            Questions per chunk
  -a, --augment-paraphrases NUM  Paraphrase variations
  --skip-generation              Skip QA generation
  --skip-training                Skip training
  --eval-only                    Only evaluate
  -h, --help                     Show help
```

---

## Command-Line Arguments (submit_pipeline.sh)

```bash
./scripts/submit_pipeline.sh [OPTIONS]

Options:
  --cpu                          Use CPU instead of GPU
  -m, --model MODEL              Model name
  -i, --input-dir DIR            Input directory
  -o, --output-dir DIR           Output directory
  --output-model-dir DIR         Model output directory
  -e, --epochs NUM               Training epochs
  -b, --batch-size NUM           Batch size
  -q, --questions NUM            Questions per chunk
  -a, --augment-paraphrases NUM  Paraphrase variations
  -h, --help                     Show help
```

---

## Monitoring Jobs

```bash
# List your jobs
squeue -u $USER

# Check specific job
squeue -j JOB_ID

# View output (real-time)
tail -f logs/pipeline_JOB_ID.out
tail -f logs/pipeline_cpu_JOB_ID.out
tail -f logs/training_JOB_ID.out
tail -f logs/inference_JOB_ID.out

# View errors
tail -f logs/pipeline_JOB_ID.err

# Cancel job
scancel JOB_ID
```

---

## Resource Requirements

| Script | Time | CPUs | RAM | GPU | Partition |
|--------|------|------|-----|-----|-----------|
| run_pipeline_slurm.sh | 24h | 8 | 32G | 1 | gpu |
| run_pipeline_slurm_cpu.sh | 48h | 16 | 64G | - | cpu |
| run_training_only_slurm.sh | 12h | 8 | 32G | 1 | gpu |
| inference.slurm | 2h | 4 | 16G | 1 | gpu |

**Note:** Adjust `#SBATCH` directives in each script based on your cluster configuration.

---

## Common Workflows

### Workflow 1: First-Time Setup
```bash
# 1. Prepare documents in data_input/
# 2. Submit pipeline
./scripts/submit_pipeline.sh

# 3. Monitor
squeue -u $USER
tail -f logs/pipeline_*.out
```

### Workflow 2: Use Existing Dataset
```bash
# Skip generation, just train
export OUTPUT_DIR="data_output"  # where dataset is
sbatch scripts/run_training_only_slurm.sh
```

### Workflow 3: Test Model on New Documents
```bash
./scripts/run_eval_on_documents.sh \
  --peft-model ./fine_tuned_weights \
  --input-dir ./new_documents
```

### Workflow 4: Interactive Inference
```bash
# Local (not SLURM)
python src/inference/inference.py \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --peft-model fine_tuned_weights
```

### Workflow 5: Custom Configuration
```bash
./scripts/submit_pipeline.sh \
  -m microsoft/phi-3-mini-4k-instruct \
  -i ./my_documents \
  -o ./my_output \
  --output-model-dir ./my_model \
  -e 15 \
  -q 5 \
  -a 3
```

---

## Troubleshooting

### Job Fails Immediately
```bash
# Check error log
cat logs/pipeline_JOB_ID.err

# Common issues:
# - Wrong partition name → Edit #SBATCH --partition
# - Not enough resources → Reduce memory/CPUs
# - Module not loaded → Add module load commands if needed
```

### Python Module Not Found
```bash
# Add to script (before python/bash calls):
module load python/3.9  # or your version
source venv/bin/activate  # if using venv
```

### Out of Memory
```bash
# Reduce batch size
export BATCH_SIZE=1

# Or request more memory in #SBATCH directives
#SBATCH --mem=64G
```

### GPU Not Available
```bash
# Use CPU script instead
./scripts/submit_pipeline.sh --cpu
```

---

## Best Practices

1. **Always use submit_pipeline.sh** for easy submission
2. **Check logs** regularly during long jobs
3. **Start small** (few epochs, small dataset) to test
4. **Use CPU mode** if GPUs are busy
5. **Set OUTPUT_MODEL_DIR** to unique names for experiments
6. **Keep data_input/** organized
7. **Monitor squeue** to check job status
8. **Adjust resources** based on your cluster limits

---

## File Outputs

### After run_pipeline.sh:
```
data_output/
  ├── qa_dataset.jsonl              # Generated QA pairs
  ├── qa_cleaned.jsonl              # Cleaned QA
  ├── conversational_data.jsonl     # Greetings, etc.
  ├── multiturn_data.jsonl          # Multi-turn conversations
  ├── training_dataset.jsonl        # Merged training data
  ├── training_train.jsonl          # Train split (80%)
  ├── training_val.jsonl            # Validation split (10%)
  ├── training_test.jsonl           # Test split (10%)
  └── evaluation_report.json        # Evaluation metrics

fine_tuned_weights/                 # Fine-tuned model
  ├── adapter_config.json
  ├── adapter_model.bin
  └── ...
```

### After run_eval_on_documents.sh:
```
eval_output/
  ├── generated_qa_eval.jsonl       # QA from new docs
  └── evaluation_report.json        # Evaluation on new docs
```

---

**For more details, see:**
- `README.md` - Full project documentation

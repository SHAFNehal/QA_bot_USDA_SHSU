# QA Dataset Generator and Fine-tuning Pipeline

A complete end-to-end Python pipeline for generating question-answer datasets from text documents and fine-tuning language models using LoRA (Low-Rank Adaptation). Build domain-specific chatbots with conversational abilities, multi-turn dialogue support, and robust question understanding.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
  - [Full Pipeline (Recommended)](#full-pipeline-recommended)
  - [Step-by-Step Instructions](#step-by-step-instructions)
- [Configuration](#configuration)
- [Data Formats](#data-formats)
- [Training Features](#training-features)
- [Supported Models](#supported-models)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)

## Features

### Core Capabilities
- **QA Dataset Generation**: Automatically extract QA pairs from text documents using LLM
- **Multi-format Support**: Process .txt, .md, and .docx files
- **Conversational Data**: Built-in greetings, farewells, gratitude, and casual conversation
- **Paraphrase Augmentation**: Generate question variations (rule-based and LLM-based)
- **Multi-turn Conversations**: Support for follow-up questions with coreference resolution
- **Data Cleaning**: Robust validation and filtering with conversational data support
- **Efficient Training**: LoRA fine-tuning with mixed precision and gradient checkpointing
- **Early Stopping**: Automatic training termination to prevent overfitting
- **Interactive Chatbot**: Run fine-tuned models with conversation history
- **Model Evaluation**: Comprehensive testing on greetings, paraphrases, and follow-ups

### Advanced Features
- Response-only loss masking (only train on assistant responses)
- Semantic duplicate detection
- Train/validation split with validation loss tracking
- Configurable hyperparameters via config.py
- Comprehensive unit test suite
- End-to-end pipeline script with model selection

## Project Structure

```
QA_bot_USDA_SHSU-version2.0/
├── src/
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── dataset_creator.py       # Generate QA pairs from documents
│   │   ├── data_cleaner.py          # Validate and filter QA pairs
│   │   ├── merge_datasets.py        # Combine all data sources
│   │   └── generators/
│   │       ├── __init__.py
│   │       ├── conversational_data.py   # Greeting/thanks/farewell pairs
│   │       ├── paraphrase_generator.py  # Question variation generation
│   │       └── multiturn_generator.py   # Multi-turn conversation data
│   ├── training/
│   │   ├── __init__.py
│   │   ├── fine_tuner.py            # LoRA fine-tuning with early stopping
│   │   └── sft.py                   # Alternative SFT trainer
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── inference.py             # Interactive chatbot interface
│   │   └── evaluate.py              # Model evaluation script
│   └── utils/
│       ├── __init__.py
│       ├── model_utils.py           # Centralized model loading
│       ├── llm_utils.py             # LLM generation utilities
│       └── file_processor.py        # Document processing
├── scripts/
│   └── run_pipeline.sh              # End-to-end pipeline script
├── tests/                           # Unit test suite
│   ├── conftest.py                  # Test fixtures
│   ├── test_conversational_data.py
│   ├── test_data_cleaner.py
│   ├── test_multiturn_generator.py
│   └── test_paraphrase_generator.py
├── config.py                        # Configuration settings
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── CLAUDE.md                        # AI assistant guidance
└── IMPLEMENTATION_PLAN.md           # Development roadmap
```

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, but recommended for faster training)
- 8GB+ RAM (16GB+ recommended)

### Install Dependencies

1. Clone the repository:
```bash
git clone <repository-url>
cd QA_bot_USDA_SHSU-version2.0
```

2. Install Python packages:
```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `torch` - PyTorch for model training
- `transformers` - Hugging Face Transformers library
- `peft` - Parameter-Efficient Fine-Tuning (LoRA)
- `datasets` - Dataset processing
- `trl` - Transformer Reinforcement Learning
- `python-docx` - DOCX file processing
- `sentence-transformers` - Semantic similarity (optional)
- `pytest` - Testing framework

## Quick Start

### 1. Prepare Your Data

Create a `data_input` directory and add your text documents:

```bash
mkdir data_input
# Add your .txt, .md, or .docx files to data_input/
```

### 2. Run the Full Pipeline

The easiest way to get started is to run the complete pipeline:

```bash
./scripts/run_pipeline.sh
```

This will:
1. Generate QA pairs from your documents
2. Clean and validate the data
3. Create multi-turn conversation examples
4. Merge all datasets
5. Fine-tune the model with LoRA
6. Save the fine-tuned model
7. Evaluate model performance

### 3. Chat with Your Model

Once training is complete, start an interactive chat session:

```bash
python src/inference/inference.py --peft_model fine_tuned_weights --interactive
```

**Interactive Commands:**
- Type questions normally to get responses
- `clear` - Clear conversation history
- `quit` or `exit` - Exit the chatbot

## Detailed Usage

### Full Pipeline (Recommended)

The `run_pipeline.sh` script provides a complete end-to-end workflow with extensive configuration options.

#### Basic Usage

```bash
# Run with default settings (TinyLlama)
./scripts/run_pipeline.sh

# Use a different model (Phi-3)
./scripts/run_pipeline.sh --model microsoft/phi-3-mini-4k-instruct --epochs 15

# Use Llama 3 with custom settings
./scripts/run_pipeline.sh --model meta-llama/Meta-Llama-3-8B-Instruct --epochs 20 --batch-size 4
```

#### Pipeline Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model MODEL` | Model to use for generation and fine-tuning | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| `-i, --input-dir DIR` | Input directory for documents | data_input |
| `-o, --output-dir DIR` | Output directory for datasets | data_output |
| `--output-model-dir DIR` | Output directory for model weights | fine_tuned_weights |
| `-e, --epochs NUM` | Number of training epochs | 10 |
| `-b, --batch-size NUM` | Training batch size | 2 |
| `-q, --questions NUM` | Questions per chunk during generation | 3 |
| `-a, --augment-paraphrases NUM` | Number of paraphrase variations | 0 |
| `--skip-generation` | Skip QA generation (use existing dataset) | false |
| `--skip-training` | Skip training step | false |
| `--eval-only` | Only run evaluation | false |
| `-h, --help` | Show help message | - |

#### Advanced Pipeline Examples

```bash
# Generate dataset with paraphrase augmentation
./scripts/run_pipeline.sh --questions 5 --augment-paraphrases 3

# Skip dataset generation and use existing data
./scripts/run_pipeline.sh --skip-generation --model microsoft/phi-3-mini-4k-instruct

# Only run evaluation on existing model
./scripts/run_pipeline.sh --eval-only

# Custom directories
./scripts/run_pipeline.sh \
    --input-dir my_docs \
    --output-dir my_output \
    --output-model-dir my_model

# High-quality dataset with more questions and variations
./scripts/run_pipeline.sh \
    --questions 10 \
    --augment-paraphrases 5 \
    --epochs 15
```

### Step-by-Step Instructions

For more control, run each stage individually:

#### 1. Generate QA Dataset

Extract QA pairs from your documents using an LLM:

```bash
python src/dataset/dataset_creator.py \
    --input_dir data_input \
    --output_file data_output/qa_dataset.jsonl \
    --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --num_questions 3
```

**Options:**
- `--input_dir`: Directory containing text files
- `--output_file`: Output JSONL file path
- `--model_name`: LLM model to use for generation
- `--num_questions`: Questions to generate per text chunk
- `--chunk_size`: Text chunk size (default: 1000)
- `--chunk_overlap`: Overlap between chunks (default: 200)
- `--augment_paraphrases`: Number of paraphrase variations per question
- `--use_llm_paraphrases`: Use LLM for paraphrase generation (slower but higher quality)
- `--advanced`: Enable advanced semantic filtering

**Examples:**

```bash
# With advanced semantic filtering
python src/dataset/dataset_creator.py \
    --input_dir data_input \
    --output_file data_output/qa_dataset.jsonl \
    --advanced

# With rule-based paraphrase augmentation
python src/dataset/dataset_creator.py \
    --input_dir data_input \
    --output_file data_output/qa_dataset.jsonl \
    --augment_paraphrases 3

# With LLM-based paraphrase augmentation (higher quality)
python src/dataset/dataset_creator.py \
    --input_dir data_input \
    --output_file data_output/qa_dataset.jsonl \
    --augment_paraphrases 3 \
    --use_llm_paraphrases
```

#### 2. Clean Dataset

Validate and filter QA pairs:

```bash
python src/dataset/data_cleaner.py \
    --input data_output/qa_dataset.jsonl \
    --output data_output/qa_cleaned.jsonl \
    --relaxed
```

**Options:**
- `--input`: Input JSONL file
- `--output`: Output JSONL file
- `--relaxed`: Use relaxed validation (allows conversational inputs)
- `--skip_duplicates`: Remove duplicate QA pairs

The cleaner will:
- Remove invalid questions (too short, malformed, etc.)
- Remove invalid answers (incomplete, too short, etc.)
- Filter out prompt leakage and artifacts
- Detect and remove duplicates (if enabled)
- Preserve conversational data (greetings, thanks, etc.)

#### 3. Generate Multi-turn Data

Create multi-turn conversation examples with coreference:

```bash
python src/dataset/generators/multiturn_generator.py \
    --input data_output/qa_cleaned.jsonl \
    --output data_output/multiturn.jsonl \
    --include_coreference
```

**Options:**
- `--input`: Input QA dataset
- `--output`: Output multi-turn dataset
- `--include_coreference`: Include predefined coreference examples
- `--num_followups`: Number of followup turns per QA pair (default: 2)

This teaches the model to:
- Handle follow-up questions
- Resolve pronouns ("it", "this", "that")
- Maintain conversation context

#### 4. Merge Datasets

Combine QA, conversational, and multi-turn data:

```bash
python src/dataset/merge_datasets.py \
    --qa_data data_output/qa_cleaned.jsonl \
    --multiturn_data data_output/multiturn.jsonl \
    --output data_output/training_dataset.jsonl \
    --include_conversational \
    --conversational_multiplier 3
```

**Options:**
- `--qa_data`: QA dataset file
- `--multiturn_data`: Multi-turn dataset file
- `--output`: Output merged dataset
- `--include_conversational`: Include greeting/farewell pairs
- `--conversational_multiplier`: Repeat conversational data N times (for better greeting responses)

#### 5. Fine-tune Model

Train the model with LoRA:

```bash
python src/training/fine_tuner.py \
    --dataset_path data_output/training_dataset.jsonl \
    --output_dir fine_tuned_weights \
    --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --early_stopping_patience 3
```

**Training Options:**
- `--dataset_path`: Path to training dataset
- `--output_dir`: Output directory for model weights
- `--model_name`: Base model to fine-tune
- `--num_train_epochs`: Number of training epochs
- `--per_device_train_batch_size`: Batch size per GPU
- `--gradient_accumulation_steps`: Accumulate gradients for larger effective batch size
- `--learning_rate`: Learning rate (default: 2e-4)
- `--early_stopping_patience`: Stop after N epochs without improvement
- `--device`: Device to use (cuda/cpu/mps)

**LoRA Configuration:**
- `--lora_r`: LoRA rank (default: 16)
- `--lora_alpha`: LoRA alpha (default: 32)
- `--lora_dropout`: LoRA dropout (default: 0.05)

#### 6. Evaluate Model

Test model performance:

```bash
python src/inference/evaluate.py \
    --model_path fine_tuned_weights \
    --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --output data_output/evaluation_report.json
```

**Evaluation Tests:**
- Greeting responses
- Paraphrased questions
- Follow-up questions
- Coreference resolution
- General QA accuracy

#### 7. Interactive Chat

Run the chatbot:

```bash
python src/inference/inference.py \
    --peft_model fine_tuned_weights \
    --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --interactive
```

**Inference Options:**
- `--peft_model`: Path to fine-tuned LoRA weights
- `--base_model`: Base model name (should match training)
- `--interactive`: Enable interactive mode
- `--max_new_tokens`: Max tokens to generate (default: 256)
- `--temperature`: Sampling temperature (default: 0.7)

**Batch Inference:**
```bash
python src/inference/inference.py \
    --peft_model fine_tuned_weights \
    --input_file questions.txt \
    --output_file answers.txt
```

## Configuration

All default settings are defined in `config.py`:

### Training Configuration
```python
DEFAULT_TRAINING_CONFIG = {
    "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "num_train_epochs": 10,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "max_seq_length": 512,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    # ... more settings
}
```

### Conversational Configuration
```python
CONVERSATIONAL_CONFIG = {
    "include_greetings": True,
    "include_farewells": True,
    "include_gratitude": True,
    "multiplier": 3  # Repeat conversational data 3x
}
```

### Paraphrase Configuration
```python
PARAPHRASE_CONFIG = {
    "variations_per_question": 3,
    "include_imperatives": True,  # "Tell me about X"
    "use_llm_paraphrases": False  # Rule-based by default
}
```

### Multi-turn Configuration
```python
MULTITURN_CONFIG = {
    "num_followups": 2,
    "include_coreference": True,
    "followup_probability": 0.3
}
```

You can override these settings via command-line arguments or by editing `config.py` directly.

## Data Formats

### TinyLlama Chat Format

The pipeline uses TinyLlama's chat format for training and inference:

```
<|system|>
You are a helpful, friendly assistant. You can answer questions, engage in conversation, and assist users with their needs.</s>
<|user|>
What is machine learning?</s>
<|assistant|>
Machine learning is a subset of artificial intelligence that enables systems to learn from data and improve their performance without being explicitly programmed.</s>
```

### JSONL Format (Between Stages)

QA pairs are stored in JSONL format:

```json
{"question": "What is machine learning?", "answer": "Machine learning is...", "source_file": "ml_intro.txt", "chunk_index": 0}
{"question": "How does deep learning work?", "answer": "Deep learning uses...", "source_file": "ml_intro.txt", "chunk_index": 1}
```

### Multi-turn Format

Multi-turn conversations include conversation history:

```json
{
  "input": "<|system|>\n...<|user|>\nWhat is AI?</s><|assistant|>\nAI is...</s><|user|>\nHow does it work?</s>",
  "output": "It works by...",
  "turn_number": 2,
  "total_turns": 3,
  "type": "multiturn"
}
```

## Training Features

### Loss Masking
Only computes loss on assistant responses (not on user messages or system prompts) using `DataCollatorForCompletionOnly`. This improves training efficiency and model quality.

### Mixed Precision Training
Automatically uses FP16 on CUDA GPUs for:
- 2x faster training
- 50% less memory usage
- Minimal accuracy impact

### Gradient Checkpointing
Reduces memory usage during training by recomputing activations during backward pass. Essential for training larger models on consumer GPUs.

### Early Stopping
Monitors validation loss and stops training when:
- No improvement for N epochs (configurable patience)
- Prevents overfitting
- Saves training time

### Train/Validation Split
- Automatically splits data 90/10
- Tracks validation loss each epoch
- Logs training metrics

### LoRA (Low-Rank Adaptation)
- Parameter-efficient fine-tuning
- Only trains ~1% of parameters
- Much faster than full fine-tuning
- Smaller model checkpoints (~5-50MB vs. GB)
- Can be merged with base model later

### Learning Rate Scheduling
Uses cosine annealing with warmup:
- Gradual warmup in first 10% of training
- Smooth decay to minimum LR
- Prevents training instability

## Supported Models

### Recommended Models

1. **TinyLlama/TinyLlama-1.1B-Chat-v1.0** (Default)
   - Best for: Getting started, limited hardware
   - Size: 1.1B parameters
   - VRAM: ~4GB
   - Speed: Fast training and inference

2. **microsoft/phi-3-mini-4k-instruct**
   - Best for: Better quality responses
   - Size: 3.8B parameters
   - VRAM: ~8GB
   - Speed: Moderate

3. **meta-llama/Meta-Llama-3-8B-Instruct**
   - Best for: Production use, highest quality
   - Size: 8B parameters
   - VRAM: ~16GB
   - Speed: Slower but best quality

### Using Different Models

```bash
# TinyLlama (default)
./scripts/run_pipeline.sh

# Phi-3
./scripts/run_pipeline.sh --model microsoft/phi-3-mini-4k-instruct

# Llama 3
./scripts/run_pipeline.sh --model meta-llama/Meta-Llama-3-8B-Instruct
```

**Note:** Larger models require more VRAM and training time but generally produce better results.

## Testing

Run the unit test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_data_cleaner.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

**Test Coverage:**
- Data cleaning and validation
- Conversational data generation
- Paraphrase generation
- Multi-turn conversation generation
- Model utilities

## Troubleshooting

### Common Issues

#### Out of Memory Errors

**Problem:** CUDA out of memory during training

**Solutions:**
1. Reduce batch size:
   ```bash
   ./scripts/run_pipeline.sh --batch-size 1
   ```

2. Increase gradient accumulation:
   ```bash
   python src/training/fine_tuner.py --per_device_train_batch_size 1 --gradient_accumulation_steps 8
   ```

3. Use CPU mode (slower):
   ```bash
   python src/training/fine_tuner.py --device cpu
   ```

4. Use a smaller model:
   ```bash
   ./scripts/run_pipeline.sh --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
   ```

#### Slow Training

**Problem:** Training takes too long

**Solutions:**
1. Use mixed precision (automatic on CUDA)
2. Reduce number of epochs:
   ```bash
   ./scripts/run_pipeline.sh --epochs 5
   ```
3. Use a smaller model
4. Enable gradient checkpointing (already enabled by default)

## Examples

### Example 1: Domain-Specific Chatbot

```bash
# 1. Add medical textbooks to data_input/
mkdir data_input
cp medical_textbook.txt data_input/

# 2. Generate high-quality dataset with paraphrases
./scripts/run_pipeline.sh \
    --model microsoft/phi-3-mini-4k-instruct \
    --questions 10 \
    --augment-paraphrases 5 \
    --epochs 15

# 3. Test the chatbot
python src/inference/inference.py \
    --peft_model fine_tuned_weights \
    --interactive
```

### Example 2: Support Bot

```bash
# 1. Add FAQ documents and support tickets
mkdir data_input
cp faq.md support_docs/*.txt data_input/

# 2. Train with emphasis on conversational ability
./scripts/run_pipeline.sh \
    --questions 5 \
    --epochs 12

# 3. Merge with extra conversational data
python src/dataset/merge_datasets.py \
    --qa_data data_output/qa_cleaned.jsonl \
    --multiturn_data data_output/multiturn.jsonl \
    --output data_output/training_dataset.jsonl \
    --include_conversational \
    --conversational_multiplier 10

# 4. Fine-tune
python src/training/fine_tuner.py \
    --dataset_path data_output/training_dataset.jsonl \
    --output_dir support_bot_weights \
    --num_train_epochs 12
```

### Example 3: Quick Prototype (Fast Mode)

```bash
# Use TinyLlama for fast training
./scripts/run_pipeline.sh \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --questions 3 \
    --epochs 5 \
    --batch-size 4
```

### Example 4: High-Quality Production Bot

```bash
# Use Llama 3 with extensive training
./scripts/run_pipeline.sh \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --questions 15 \
    --augment-paraphrases 7 \
    --epochs 20 \
    --batch-size 2
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass (`pytest tests/`)
5. Submit a pull request

## License

This project is for research and educational purposes. Please check individual model licenses before commercial use.


## Acknowledgments

- Built with [Hugging Face Transformers](https://github.com/huggingface/transformers)
- Uses [PEFT](https://github.com/huggingface/peft) for LoRA implementation
- Inspired by [TRL](https://github.com/huggingface/trl) for training utilities
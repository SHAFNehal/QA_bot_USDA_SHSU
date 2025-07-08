### An LLM was used to create this documentation.

# QA Dataset Generator and Fine-tuning Pipeline

A complete Python pipeline for generating question-answer datasets from text documents and fine-tuning language models using LoRA (Low-Rank Adaptation).

## Overview

This project provides a streamlined workflow to:
1. **Generate QA datasets** from text documents (.txt, .md, .docx)
2. **Clean and validate** the generated data
3. **Fine-tune language models** using LoRA for efficient training
4. **Run interactive QA chatbot** with fine-tuned models

## Project Structure

```
llm_qa_pipeline/
├── data_input/                 # Input documents (.txt, .md, .docx)
├── data_output/               # Generated QA datasets
├── fine_tuned_weights/        # Saved fine-tuned model weights
├── utils/
│   ├── file_processor.py     # Document processing utilities
│   └── llm_utils.py          # LLM utilities for QA generation
├── docs/
│   └── qa_generation_guide.md # Detailed guide for QA generation
├── dataset_creator.py         # Main script to generate QA datasets
├── data_cleaner.py           # Clean and validate generated data
├── fine_tuner.py             # Fine-tune models using LoRA
├── inference.py              # Interactive QA chatbot
├── qa_calculator.py          # Calculate optimal QA pair counts
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── QUICK_START.md           # Quick start guide
└── PROJECT_STRUCTURE.md     # Detailed project structure
```

## Installation

```bash
# Install required dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Your Data
Place your documents in `data_input/`:
- Supported formats: `.txt`, `.md`, `.docx`
- Can be organized in subdirectories

### 2. Generate QA Dataset
```bash
python dataset_creator.py \
    --input_dir data_input \
    --output_file data_output/qa_dataset.jsonl \
    --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --chunk_size 1000 \
    --overlap 200 \
    --num_questions 3
```

### 3. Clean the Dataset
```bash
python data_cleaner.py \
    --input data_output/qa_dataset.jsonl \
    --output data_output/qa_dataset_cleaned.jsonl
```

### 4. Fine-tune the Model
```bash
echo "Step 4: Fine-tuning model..."
python fine_tuner.py \
    --dataset_path data_output/qa_dataset_cleaned.jsonl \
    --output_dir fine_tuned_weights \
    --num_train_epochs 10 \
    --per_device_train_batch_size 2 \
    --learning_rate 3e-5 \
    --warmup_steps 100 \
    --logging_steps 20 \
    --save_steps 200
```

### 5. Run QA Chatbot
```bash
python inference.py \
    --peft_model fine_tuned_weights \
    --interactive
```

## Core Components

### Dataset Generation (`dataset_creator.py`)
- Processes text documents into chunks
- Generates QA pairs using LLMs (TinyLlama, Phi-3 Mini, LLaMA 3.1)
- Supports multiple output formats
- Configurable chunk size and overlap

**Key Features:**
- Multi-format document support
- Configurable chunking strategy
- Multiple LLM backends
- Progress tracking and logging

### Data Cleaning (`data_cleaner.py`)
- Removes malformed QA pairs
- Filters incomplete answers
- Removes duplicates
- Validates question format

**Cleaning Steps:**
- Question validation (must end with ?)
- Answer completeness check
- Duplicate detection
- Format artifact removal

### Fine-tuning (`fine_tuner.py`)
- LoRA-based parameter-efficient training
- Support for multiple base models
- Automatic device detection
- Simple configuration

**Training Features:**
- LoRA integration (r=16, alpha=32)
- FP16 mixed precision training
- Automatic model saving
- Progress logging
- **Configurable hyperparameters:**
    - `--num_train_epochs` (default: 20)
    - `--per_device_train_batch_size` (default: 4)
    - `--learning_rate` (default: 5e-5)
    - `--warmup_steps` (default: 50)
    - `--logging_steps` (default: 10)
    - `--save_steps` (default: 500)

### Interactive Chatbot (`inference.py`)
- Loads fine-tuned models
- Interactive Q&A sessions
- Batch inference support
- Configurable generation parameters

**Usage Modes:**
- Interactive chat mode
- Batch question processing
- Custom generation parameters

## Configuration

### Model Options
- **TinyLlama/TinyLlama-1.1B-Chat-v1.0** (default, recommended)
- **microsoft/Phi-3-mini-4k-instruct**
- **meta-llama/Meta-Llama-3.1-8B-Instruct**

### Generation Parameters
- `--chunk_size`: Text chunk size (default: 1000)
- `--overlap`: Chunk overlap (default: 200)
- `--num_questions`: Questions per chunk (default: 3)
- `--temperature`: Generation temperature (default: 0.7)
- `--max_new_tokens`: Max tokens to generate (default: 256)

### Training Parameters
- `--learning_rate`: Learning rate (default: 5e-5)
- `--num_epochs`: Training epochs (default: 3)
- `--batch_size`: Batch size (default: 4)
- `--warmup_steps`: Warmup steps (default: 50)

## Advanced Usage

### QA Calculator
Calculate optimal number of QA pairs for your use case:
```bash
python qa_calculator.py --interactive
```

### Custom Configuration
Modify `config.py` for project-wide settings:
- Model configurations
- Generation parameters
- File paths
- Device settings

### Batch Processing
Process multiple documents:
```bash
python dataset_creator.py --input_dir data_input --batch_size 10
```

## Troubleshooting

### Memory Issues
- Reduce batch size: `--batch_size 2`
- Use smaller chunk size: `--chunk_size 500`
- Enable CPU mode: `--device cpu`

### Poor QA Quality
- Increase chunk size for more context
- Reduce questions per chunk
- Try different base models
- Use data cleaner to filter results

### Training Issues
- Check dataset quality with data cleaner
- Reduce learning rate
- Increase warmup steps
- Verify LoRA configuration

## Expected Results

### Dataset Generation
- **Raw dataset**: 1000-5000 QA pairs (depending on input)
- **Cleaned dataset**: 60-80% of raw data (after filtering)
- **Processing time**: 10-30 minutes (depending on document size)

### Fine-tuning
- **Training time**: 1-3 hours on GPU for 3 epochs
- **Model size**: ~50MB LoRA weights (vs 2GB+ full model)
- **Memory usage**: 4-8GB GPU memory

### Inference
- **Response time**: 1-5 seconds per question
- **Quality**: Improved domain-specific answers
- **Memory**: 2-4GB GPU memory

## File Descriptions

### Core Scripts
- `dataset_creator.py`: Main QA generation script
- `data_cleaner.py`: Data validation and cleaning
- `fine_tuner.py`: LoRA fine-tuning script
- `inference.py`: Interactive chatbot

### Utilities
- `utils/file_processor.py`: Document processing utilities
- `utils/llm_utils.py`: LLM integration and generation
- `qa_calculator.py`: QA pair count calculator
- `config.py`: Configuration management

### Documentation
- `README.md`: Main documentation (this file)
- `QUICK_START.md`: Quick start guide
- `PROJECT_STRUCTURE.md`: Detailed project structure
- `docs/qa_generation_guide.md`: QA generation best practices

## Dependencies

Key packages:
- `transformers`: Hugging Face transformers
- `torch`: PyTorch
- `peft`: Parameter-efficient fine-tuning
- `datasets`: Dataset handling
- `docx2txt`: Word document processing
- `markdown`: Markdown processing

## Contributing

1. Follow the existing code structure
2. Add proper error handling
3. Include documentation for new features
4. Test with different model configurations

## License

This project is for research and educational purposes.
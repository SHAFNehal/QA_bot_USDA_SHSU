"""
Configuration file for the LLM QA Pipeline

This file contains default settings and configurations for the pipeline.
"""

# Default model configurations
DEFAULT_MODELS = {
    "phi3_mini": {
        "name": "microsoft/phi-3-mini-4k-instruct",
        "max_length": 1024,
        "temperature": 0.7,
        "description": "3.8B parameters, instruction-tuned"
    },
    "llama3_1": {
        "name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "max_length": 1024,
        "temperature": 0.7,
        "description": "8B parameters, latest Llama 3.1 model"
    },
    "tinyllama": {
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "max_length": 512,
        "temperature": 0.8,
        "description": "1.1B parameters, very fast"
    }
}

# Default training configurations
DEFAULT_TRAINING_CONFIG = {
    "num_epochs": 10,
    "batch_size": 4,
    "learning_rate": 2e-5,
    "warmup_steps": 100,
    "max_seq_length": 512,
    "gradient_accumulation_steps": 1,
    "save_steps": 500,
    "eval_steps": 500,
    "logging_steps": 10
}

# Default dataset creation configurations
DEFAULT_DATASET_CONFIG = {
    "chunk_size": 1000,
    "overlap": 200,
    "num_questions_per_chunk": 10, # << This is where you can change the number of questions per chunk
    "temperature": 0.7,
    "max_length": 1024
}

# File processing configurations
FILE_PROCESSING_CONFIG = {
    "supported_extensions": [".txt", ".md", ".docx"],
    "encoding": "utf-8",
    "fallback_encoding": "latin-1",
    "min_chunk_size": 100,
    "max_chunk_size": 2000
}

# Output formats
OUTPUT_FORMATS = {
    "instruction": {
        "template": "### Instruction:\n{question}\n\n### Response:\n{answer}",
        "description": "Instruction-following format"
    },
    "chat": {
        "template": "<|user|>\n{question}\n<|assistant|>\n{answer}",
        "description": "Chat format with special tokens"
    },
    "simple": {
        "template": "Question: {question}\nAnswer: {answer}",
        "description": "Simple Q&A format"
    }
}

# Quality control settings
QUALITY_CONTROL = {
    "min_question_length": 10,
    "min_answer_length": 20,
    "max_question_length": 200,
    "max_answer_length": 500,
    "remove_duplicates": True,
    "filter_empty": True
}

# Device configurations
DEVICE_CONFIG = {
    "auto": "Automatically detect best device",
    "cpu": "Force CPU usage",
    "cuda": "Force CUDA usage"
}

def get_model_config(model_key: str = "tinyllama"):
    """Get configuration for a specific model."""
    return DEFAULT_MODELS.get(model_key, DEFAULT_MODELS["tinyllama"])

def get_training_config(**overrides):
    """Get training configuration with optional overrides."""
    config = DEFAULT_TRAINING_CONFIG.copy()
    config.update(overrides)
    return config

def get_dataset_config(**overrides):
    """Get dataset creation configuration with optional overrides."""
    config = DEFAULT_DATASET_CONFIG.copy()
    config.update(overrides)
    return config

def get_output_format(format_type: str = "instruction"):
    """Get output format configuration."""
    return OUTPUT_FORMATS.get(format_type, OUTPUT_FORMATS["instruction"])

def list_available_models():
    """List all available models with descriptions."""
    models = []
    for key, config in DEFAULT_MODELS.items():
        models.append({
            "key": key,
            "name": config["name"],
            "description": config["description"]
        })
    return models

def validate_config(config: dict, config_type: str = "training"):
    """Validate configuration settings."""
    if config_type == "training":
        required_keys = ["num_epochs", "batch_size", "learning_rate"]
    elif config_type == "dataset":
        required_keys = ["chunk_size", "overlap", "num_questions_per_chunk"]
    else:
        return True
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    return True 
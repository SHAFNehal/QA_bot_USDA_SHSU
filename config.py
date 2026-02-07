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
    "batch_size": 2,
    "learning_rate": 5e-5,
    "warmup_steps": 100,
    "max_seq_length": 2048,  # Increased for multi-turn support
    "gradient_accumulation_steps": 4,  # Effective batch size = batch_size * 4
    "save_steps": 200,
    "eval_steps": 200,
    "logging_steps": 20,
    "early_stopping_patience": 3,
    "lr_scheduler_type": "cosine",
    "fp16": True,  # Mixed precision (auto-disabled on CPU)
    "gradient_checkpointing": True,
    "train_val_split": 0.1,  # 10% validation
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

# Conversational data settings
CONVERSATIONAL_CONFIG = {
    "include_greetings": True,
    "include_farewells": True,
    "include_gratitude": True,
    "include_acknowledgments": True,
    "include_meta_questions": True,
    "conversational_multiplier": 3,  # Repeat conversational data N times
}

# Multi-turn conversation settings
MULTITURN_CONFIG = {
    "max_history_turns": 5,
    "include_coreference_examples": True,
}

# Paraphrase augmentation settings
PARAPHRASE_CONFIG = {
    "variations_per_question": 3,
    "include_imperatives": True,  # "Tell me about X" style
}

# System prompt for training and inference
SYSTEM_PROMPT = "You are a helpful, friendly assistant. You can answer questions, engage in conversation, and assist users with their needs."

# Device configurations
DEVICE_CONFIG = {
    "auto": "Automatically detect best device",
    "cpu": "Force CPU usage",
    "cuda": "Force CUDA usage"
}

# RAG (Retrieval-Augmented Generation) configuration
RAG_CONFIG = {
    "top_k": 5,
    "chunk_size": FILE_PROCESSING_CONFIG.get("max_chunk_size", 2000) // 2,  # 1000 default
    "overlap": FILE_PROCESSING_CONFIG.get("min_chunk_size", 100),
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "use_hybrid": False,
    "collection_name": "rag_docs",
}

def get_rag_config(**overrides):
    """Get RAG configuration with optional overrides."""
    config = RAG_CONFIG.copy()
    config.update(overrides)
    return config

# Hybrid RAG (restate -> RAG + FT -> synthesize) configuration
HYBRID_RAG_CONFIG = {
    "small_llm_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "num_restatements": 5,
    "per_query_top_k": 5,
    "merged_retrieval_top_k": 12,
    "max_new_tokens_restater": 256,
    "max_new_tokens_synthesizer": 512,
    "temperature": 0.7,
}

def get_hybrid_rag_config(**overrides):
    """Get Hybrid RAG configuration with optional overrides."""
    config = HYBRID_RAG_CONFIG.copy()
    config.update(overrides)
    return config

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
"""
Centralized Model Loading Utilities

This module provides a unified interface for loading models and tokenizers
across all scripts in the project (training, inference, dataset generation).
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, Optional, Literal


# System prompt used across training and inference
SYSTEM_PROMPT = "You are a helpful, friendly assistant. You can answer questions, engage in conversation, and assist users with their needs."


def get_device() -> str:
    """Determine the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_tokenizer(
    model_name: str,
    device: Optional[str] = None,
    for_training: bool = False,
    enable_gradient_checkpointing: bool = True,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model and tokenizer with appropriate settings for training or inference.

    Args:
        model_name: HuggingFace model name or local path
        device: Device to use ('cuda', 'cpu', 'mps', or None for auto)
        for_training: If True, prepare model for training (gradient checkpointing, etc.)
        enable_gradient_checkpointing: Enable gradient checkpointing for memory efficiency
        load_in_8bit: Load model in 8-bit precision (requires bitsandbytes)
        load_in_4bit: Load model in 4-bit precision (requires bitsandbytes)

    Returns:
        Tuple of (model, tokenizer)
    """
    # Determine device
    if device is None:
        device = get_device()

    print(f"Loading model: {model_name}")
    print(f"Device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Determine dtype and device_map
    # For training: use float32 and let TrainingArguments handle FP16
    # For inference: use float16 for efficiency
    if device == "cuda":
        if for_training:
            # Training: use float32, let TrainingArguments handle mixed precision
            dtype = torch.float32
            device_map = None  # Don't use device_map for training
        else:
            # Inference: use float16 for efficiency
            dtype = torch.float16
            device_map = "auto"
    else:
        dtype = torch.float32
        device_map = None

    # Build model loading kwargs
    model_kwargs = {
        "dtype": dtype,  # Changed from torch_dtype to dtype (deprecated)
        "device_map": device_map,
        "trust_remote_code": True,
    }

    # Add quantization if requested
    if load_in_8bit:
        model_kwargs["load_in_8bit"] = True
        model_kwargs.pop("dtype", None)  # Changed from torch_dtype to dtype
    elif load_in_4bit:
        model_kwargs["load_in_4bit"] = True
        model_kwargs.pop("dtype", None)  # Changed from torch_dtype to dtype

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Move to device if no device_map
    if device_map is None and not (load_in_8bit or load_in_4bit):
        model = model.to(device)
    
    # For training with FP16, ensure model is on CUDA before enabling mixed precision
    if for_training and device == "cuda" and not (load_in_8bit or load_in_4bit):
        # Model should be in float32 for training, TrainingArguments will handle FP16
        if model.dtype != torch.float32:
            model = model.float()

    # Training-specific setup
    if for_training:
        # Enable gradient checkpointing for memory efficiency
        if enable_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")

        # Ensure model is in training mode
        model.train()
    else:
        # Inference mode
        model.eval()

    return model, tokenizer


def load_peft_model(
    base_model_name: str,
    peft_model_path: str,
    device: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a PEFT/LoRA fine-tuned model.

    Args:
        base_model_name: HuggingFace model name for the base model
        peft_model_path: Path to the PEFT adapter weights
        device: Device to use (None for auto)

    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        from peft import PeftModel
    except ImportError:
        raise ImportError("Please install peft: pip install peft")

    # Determine device
    if device is None:
        device = get_device()

    print(f"Loading base model: {base_model_name}")
    print(f"Loading PEFT adapter: {peft_model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load base model
    dtype = torch.float16 if device == "cuda" else torch.float32
    device_map = "auto" if device == "cuda" else None

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=dtype,  # Changed from torch_dtype to dtype (deprecated)
        device_map=device_map,
        trust_remote_code=True,
    )

    # Load PEFT adapter
    model = PeftModel.from_pretrained(base_model, peft_model_path)
    model.eval()

    return model, tokenizer


def format_chat_prompt(
    user_message: str,
    system_prompt: str = SYSTEM_PROMPT,
    conversation_history: Optional[list] = None,
) -> str:
    """
    Format a message using TinyLlama chat template.

    Args:
        user_message: The user's current message
        system_prompt: System prompt to use
        conversation_history: List of previous turns [{'user': ..., 'assistant': ...}, ...]

    Returns:
        Formatted prompt string
    """
    prompt = f"<|system|>\n{system_prompt}</s>\n"

    # Add conversation history if provided
    if conversation_history:
        for turn in conversation_history:
            prompt += f"<|user|>\n{turn['user']}</s>\n"
            prompt += f"<|assistant|>\n{turn['assistant']}</s>\n"

    # Add current user message
    prompt += f"<|user|>\n{user_message}</s>\n<|assistant|>\n"

    return prompt


def format_for_training(
    question: str,
    answer: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """
    Format a Q&A pair for training using TinyLlama chat template.

    Args:
        question: The question/input
        answer: The answer/response
        system_prompt: System prompt to use

    Returns:
        Formatted training string
    """
    return f"<|system|>\n{system_prompt}</s>\n<|user|>\n{question}</s>\n<|assistant|>\n{answer}</s>"


def get_model_info(model_name: str) -> dict:
    """
    Get information about a model.

    Args:
        model_name: HuggingFace model name

    Returns:
        Dictionary with model information
    """
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        return {
            "name": model_name,
            "hidden_size": getattr(config, 'hidden_size', None),
            "num_layers": getattr(config, 'num_hidden_layers', None),
            "num_heads": getattr(config, 'num_attention_heads', None),
            "vocab_size": getattr(config, 'vocab_size', None),
            "model_type": getattr(config, 'model_type', None),
        }
    except Exception as e:
        return {"name": model_name, "error": str(e)}

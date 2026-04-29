"""
Centralized Model Loading Utilities

This module provides a unified interface for loading models and tokenizers
across all scripts in the project (training, inference, dataset generation).
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Tuple, Optional, Literal, List, Dict


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

    # Load tokenizer (fix_mistral_regex for Mistral/Nemo; use_fast for Llama to avoid SentencePiece vocab_file TypeError)
    tok_kwargs = {"trust_remote_code": True}
    if "mistral" in model_name.lower() or "Mistral" in model_name:
        tok_kwargs["fix_mistral_regex"] = True
    if "llama" in model_name.lower() or "Llama" in model_name:
        tok_kwargs["use_fast"] = True  # Avoid slow tokenizer vocab_file TypeError when loading from local path
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Determine dtype and device_map
    #
    # NOTE:
    # For training on CUDA, keeping the base model in float32 can easily OOM even on large GPUs,
    # especially for bigger models. Using fp16/bf16 weights is standard for LoRA fine-tuning and
    # dramatically reduces VRAM without breaking correctness (Trainer still controls AMP/loss scaling).
    # For quantized (4/8-bit) loads we use device_map="auto" so the model stays on GPU(s); with
    # multiple GPUs it will split; with one GPU everything must fit (e.g. Mixtral 8x22B 4-bit ~44GB+).
    if device == "cuda":
        if for_training and not (load_in_8bit or load_in_4bit):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            device_map = None
        else:
            dtype = torch.float16
            device_map = "auto" if (load_in_8bit or load_in_4bit) else "auto"
    else:
        dtype = torch.float32
        device_map = None

    # Build model loading kwargs
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": device_map,
        "trust_remote_code": True,
    }

    # Quantization: use BitsAndBytesConfig (load_in_4bit/load_in_8bit are deprecated)
    if load_in_8bit or load_in_4bit:
        try:
            import bitsandbytes  # noqa: F401
        except Exception as e:
            raise ImportError(
                "Quantized loading (--load_in_8bit/--load_in_4bit) requires the 'bitsandbytes' package, "
                "but it is not installed (or failed to import) in this environment.\n\n"
                "Fix (inside your venv on the cluster):\n"
                "  pip install -U bitsandbytes\n\n"
                "If your cluster blocks internet installs, ask admins for a module/wheel, or disable "
                "quantization (but note: very large models like 70B will OOM in fp16 on a single 80GB GPU)."
            ) from e

        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if load_in_8bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=False,
            )
        else:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        model_kwargs.pop("torch_dtype", None)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Move to device if no device_map
    if device_map is None and not (load_in_8bit or load_in_4bit):
        model = model.to(device)
    
    # For training on CUDA we keep fp16/bf16 weights (see dtype selection above).

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

    # Load tokenizer (fix_mistral_regex for Mistral/Nemo; use_fast for Llama to avoid SentencePiece vocab_file TypeError)
    tok_kwargs = {"trust_remote_code": True}
    if "mistral" in base_model_name.lower() or "Mistral" in base_model_name:
        tok_kwargs["fix_mistral_regex"] = True
    if "llama" in base_model_name.lower() or "Llama" in base_model_name:
        tok_kwargs["use_fast"] = True
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, **tok_kwargs)
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


def format_chat_prompt_with_template(
    tokenizer: AutoTokenizer,
    user_message: str,
    system_prompt: str = SYSTEM_PROMPT,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Format a chat prompt using the tokenizer's native chat template (if available).

    This is required for some models (e.g., Llama 3.1) where the correct chat format
    is not the legacy <|user|>/<|assistant|> format.
    """
    if not hasattr(tokenizer, "apply_chat_template") or not getattr(tokenizer, "chat_template", None):
        raise ValueError("Tokenizer has no chat_template; cannot format with template.")

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if conversation_history:
        for turn in conversation_history:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})
    messages.append({"role": "user", "content": user_message})

    # add_generation_prompt=True => ends at assistant slot
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def format_chat_prompt_llama2(
    user_message: str,
    system_prompt: str = SYSTEM_PROMPT,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Format a prompt using the Llama 2 chat template.

    This is useful when tokenizer.chat_template is missing but you still want correct
    Llama 2 chat formatting.
    """
    # Llama2 chat template:
    # <s>[INST] <<SYS>>...<</SYS>>\n\nUser [/INST] Assistant </s><s>[INST] User2 [/INST]
    sys_block = f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n"

    parts: List[str] = []

    if conversation_history:
        # First turn includes system prompt
        first = True
        for turn in conversation_history:
            u = turn["user"].strip()
            a = turn["assistant"].strip()
            if first:
                parts.append(f"<s>[INST] {sys_block}{u} [/INST] {a} </s>")
                first = False
            else:
                parts.append(f"<s>[INST] {u} [/INST] {a} </s>")
    # Current user prompt (generation prompt)
    if not conversation_history:
        parts.append(f"<s>[INST] {sys_block}{user_message.strip()} [/INST]")
    else:
        parts.append(f"<s>[INST] {user_message.strip()} [/INST]")

    return "".join(parts)


def format_for_training_llama2(
    question: str,
    answer: str,
    system_prompt: str = SYSTEM_PROMPT,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Format a (question, answer) training example using the Llama 2 chat template.
    """
    prompt = format_chat_prompt_llama2(
        user_message=question,
        system_prompt=system_prompt,
        conversation_history=conversation_history,
    )
    # Append the assistant answer and close the turn.
    return f"{prompt} {answer.strip()} </s>"


def format_for_training_with_template(
    tokenizer: AutoTokenizer,
    question: str,
    answer: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """
    Format a (question, answer) pair for training using tokenizer.chat_template.
    """
    if not hasattr(tokenizer, "apply_chat_template") or not getattr(tokenizer, "chat_template", None):
        raise ValueError("Tokenizer has no chat_template; cannot format for training with template.")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


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

"""
Fine-Tuning Script for LLM QA Models

This script fine-tunes a language model on QA datasets using the TinyLlama chat format
to ensure consistency between training and inference.

Features:
- Unified chat format matching inference
- Proper loss masking (only compute loss on response tokens)
- Mixed precision training (fp16)
- Gradient checkpointing for memory efficiency
- Train/validation split with early stopping
"""

import os
import json
import argparse
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datasets import Dataset
from transformers import (
    TrainingArguments,
    Trainer,
    PreTrainedTokenizerBase,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from src.utils.model_utils import load_model_and_tokenizer, SYSTEM_PROMPT, format_for_training, format_for_training_llama2, format_chat_prompt_llama2


@dataclass
class DataCollatorForCausalLMWithLabels:
    """
    Pads input_ids/attention_mask and pads labels with -100.

    Use this when the dataset already contains completion-only labels.
    """
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        input_ids = []
        attention_mask = []
        labels = []

        for f in features:
            ids = f["input_ids"]
            mask = f.get("attention_mask", [1] * len(ids))
            lab = f["labels"]

            pad_amount = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_amount)
            attention_mask.append(mask + [0] * pad_amount)
            labels.append(lab + [-100] * pad_amount)

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

        if torch.all(batch["labels"] == -100):
            raise RuntimeError("All labels are -100 after masking; check dataset construction.")

        return batch


def _enable_input_require_grads(model) -> None:
    """
    Ensure at least one model input requires grad.

    This is critical for PyTorch gradient checkpointing to work when only adapter
    (LoRA) weights are trainable; otherwise checkpointing can produce:
      "None of the inputs have requires_grad=True. Gradients will be None"
    and backward may fail with:
      "element 0 of tensors does not require grad..."
    """
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        return

    # Fallback for older model classes: force embedding outputs to require grad.
    try:
        embeddings = model.get_input_embeddings()
    except Exception:
        embeddings = None

    if embeddings is None:
        return

    def _make_output_require_grad(_module, _inputs, output):
        try:
            output.requires_grad_(True)
        except Exception:
            pass

    embeddings.register_forward_hook(_make_output_require_grad)


@dataclass
class DataCollatorForCompletionOnly:
    """
    Custom data collator that masks prompt tokens for completion-only training.

    This ensures loss is only computed on the response/answer tokens,
    not on the prompt/question tokens.
    """
    tokenizer: PreTrainedTokenizerBase
    response_template: str = "<|assistant|>"
    mlm: bool = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # First, pad the sequences
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )

        # Create labels - copy input_ids
        labels = batch["input_ids"].clone()

        # Find the response template tokens.
        #
        # IMPORTANT:
        # - SentencePiece tokenizers (e.g., Mistral/Llama) are sensitive to leading whitespace/newlines.
        # - Many of our training strings contain "\n<|assistant|>\n" (marker at start of a new line).
        # - Encoding "<|assistant|>" in isolation often produces *different* token IDs than encoding it
        #   when preceded by "\n" or other whitespace.
        #
        # So we search multiple leading/trailing variants to robustly locate the assistant boundary.
        response_token_variants = [
            self.response_template,
            self.response_template + "\n",
            self.response_template + "\r\n",
            "\n" + self.response_template,
            "\n" + self.response_template + "\n",
            "\r\n" + self.response_template,
            "\r\n" + self.response_template + "\n",
            "</s>\n" + self.response_template,
            "</s>\n" + self.response_template + "\n",
        ]
        # Deduplicate while preserving order
        seen = set()
        response_token_ids_variants = []
        for v in response_token_variants:
            if v in seen:
                continue
            seen.add(v)
            ids = self.tokenizer.encode(v, add_special_tokens=False)
            if ids:
                response_token_ids_variants.append(ids)

        # Mask everything before the response template with -100
        for i in range(len(labels)):
            input_ids = batch["input_ids"][i].tolist()

            # Find the position of response template (any variant)
            response_start = None
            for response_token_ids in response_token_ids_variants:
                if not response_token_ids:
                    continue
                for j in range(len(input_ids) - len(response_token_ids) + 1):
                    if input_ids[j:j + len(response_token_ids)] == response_token_ids:
                        response_start = j + len(response_token_ids)
                        break
                if response_start is not None:
                    break

            # If found, mask everything before it
            if response_start is not None:
                labels[i, :response_start] = -100
            else:
                # Fallback: don't crash the entire run on a single bad/odd example.
                # We still mask padding tokens below. This means the model will learn on the entire
                # sequence for this example (prompt+completion), which is suboptimal but better than
                # hard-failing mid-training.
                #
                # If the sequence is effectively empty (e.g., all padding), we *do* crash because
                # loss would be undefined.
                decoded = self.tokenizer.decode(batch["input_ids"][i], skip_special_tokens=False)
                print(
                    "WARNING: Could not find the assistant response template in a training example; "
                    "falling back to unmasked loss for this example.\n"
                    f"Searched variants: {response_token_variants}\n"
                    f"Decoded example (truncated): {decoded[:500]}"
                )

            # Also mask padding tokens
            if self.tokenizer.pad_token_id is not None:
                labels[i][labels[i] == self.tokenizer.pad_token_id] = -100

        batch["labels"] = labels
        
        # Note: Trainer will automatically move batch to model's device
        # No need to manually move tensors here

        # Sanity check: ensure we have at least one supervised token in the batch.
        if torch.all(batch["labels"] == -100):
            raise RuntimeError(
                "All labels are -100 after masking; loss would be undefined (0/NaN). "
                "Check your chat formatting vs response_template."
            )
        
        return batch


def load_dataset(dataset_path, tokenizer, max_length=1024):
    """
    Load and format QA dataset for training.
    Supports both single-turn (question/answer) and multi-turn (input/output) formats.
    """
    print(f"Loading dataset from: {dataset_path}")

    examples = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                examples.append(item)
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(examples)} examples")

    formatted_texts = []
    for example in examples:
        # Support multi-turn format (input/output)
        if 'input' in example and 'output' in example:
            text = f"<|system|>\n{SYSTEM_PROMPT}</s>\n{example['input']}{example['output']}</s>"
        # Support single-turn format (question/answer)
        elif 'question' in example and 'answer' in example:
            question = example['question'].strip()
            answer = example['answer'].strip()
            text = format_for_training(question, answer)
        else:
            continue
        formatted_texts.append(text)

    print(f"Formatted {len(formatted_texts)} training examples")

    dataset = Dataset.from_dict({"text": formatted_texts})

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            # Let the collator pad dynamically; padding-to-max-length here can obscure
            # debugging and inflates batches with EOS-as-PAD.
            padding=False,
            max_length=max_length
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    return tokenized_dataset


def _parse_legacy_multiturn_input(input_text: str) -> List[Dict[str, Optional[str]]]:
    """
    Parse legacy multiturn input built with <|user|> ... </s> <|assistant|> ... </s>.

    Returns a list of turns. The last turn has assistant=None (current user prompt).
    """
    text = input_text or ""
    turns: List[Dict[str, Optional[str]]] = []

    user_tag = "<|user|>"
    asst_tag = "<|assistant|>"
    end_tag = "</s>"

    i = 0
    while True:
        u = text.find(user_tag, i)
        if u == -1:
            break
        u_start = u + len(user_tag)
        u_end = text.find(end_tag, u_start)
        if u_end == -1:
            break
        user_msg = text[u_start:u_end].strip()

        a = text.find(asst_tag, u_end)
        if a == -1:
            turns.append({"user": user_msg, "assistant": None})
            break
        a_start = a + len(asst_tag)
        a_end = text.find(end_tag, a_start)
        if a_end == -1:
            turns.append({"user": user_msg, "assistant": None})
            break

        assistant_msg = text[a_start:a_end].strip()
        if assistant_msg:
            turns.append({"user": user_msg, "assistant": assistant_msg})
            i = a_end + len(end_tag)
            continue

        turns.append({"user": user_msg, "assistant": None})
        break

    return turns


def load_dataset_with_chat_template(dataset_path: str, tokenizer, max_length: int = 1024) -> Dataset:
    """
    Load dataset but build input_ids/labels using tokenizer.chat_template.

    This is required for Llama 3.1 (and recommended for any model with a defined chat template).
    """
    if not hasattr(tokenizer, "apply_chat_template") or not getattr(tokenizer, "chat_template", None):
        raise RuntimeError(
            "Tokenizer has no chat_template. "
            "For Llama 3.1 you need a recent transformers version and the official tokenizer."
        )

    examples: List[Dict[str, Any]] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    rows: List[Dict[str, Any]] = []

    def build_labels(messages_wo_assistant: List[Dict[str, str]], assistant_text: str) -> Optional[Dict[str, Any]]:
        prompt_ids = tokenizer.apply_chat_template(messages_wo_assistant, tokenize=True, add_generation_prompt=True)
        full_ids = tokenizer.apply_chat_template(
            messages_wo_assistant + [{"role": "assistant", "content": assistant_text}],
            tokenize=True,
            add_generation_prompt=False,
        )
        if not isinstance(prompt_ids, list) or not isinstance(full_ids, list):
            return None

        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        attn = [1] * len(full_ids)

        # Truncate from left; keep end (assistant completion)
        if len(full_ids) > max_length:
            full_ids = full_ids[-max_length:]
            labels = labels[-max_length:]
            attn = attn[-max_length:]

        if all(x == -100 for x in labels):
            return None

        return {"input_ids": full_ids, "attention_mask": attn, "labels": labels}

    for ex in examples:
        # Multi-turn legacy input/output
        if "input" in ex and "output" in ex:
            turns = _parse_legacy_multiturn_input(str(ex["input"]))
            if not turns:
                continue

            history_msgs: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
            current_user = None
            for t in turns:
                if t["assistant"] is None:
                    current_user = t["user"]
                    break
                history_msgs.append({"role": "user", "content": t["user"]})
                history_msgs.append({"role": "assistant", "content": t["assistant"]})

            if not current_user:
                continue

            history_msgs.append({"role": "user", "content": current_user})
            built = build_labels(history_msgs, str(ex["output"]))
            if built is not None:
                rows.append(built)
            continue

        # Single-turn question/answer
        if "question" in ex and "answer" in ex:
            q = str(ex["question"]).strip()
            a = str(ex["answer"]).strip()
            if not q or not a:
                continue
            msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ]
            built = build_labels(msgs, a)
            if built is not None:
                rows.append(built)
            continue

    print(f"Loaded {len(examples)} raw examples; prepared {len(rows)} tokenized examples using chat_template")
    return Dataset.from_list(rows)


def load_dataset_with_chat_format(
    dataset_path: str,
    tokenizer,
    max_length: int = 1024,
    chat_format: str = "auto",
) -> Dataset:
    """
    Build completion-only labels using a specified chat format.

    chat_format:
      - auto: use tokenizer.chat_template (required for Llama 3.1)
      - llama2: use Llama 2 [INST] formatting (for Llama 2 chat models when chat_template is missing)
    """
    if chat_format == "auto":
        return load_dataset_with_chat_template(dataset_path, tokenizer, max_length=max_length)

    if chat_format != "llama2":
        raise ValueError(f"Unsupported chat_format: {chat_format}")

    # Llama 2 formatting path (string-based, then tokenize prompt vs full)
    examples: List[Dict[str, Any]] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    rows: List[Dict[str, Any]] = []

    def tokenize_pair(prompt_text: str, full_text: str) -> Optional[Dict[str, Any]]:
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full = tokenizer(full_text, add_special_tokens=False)
        full_ids = full["input_ids"]
        attn = full.get("attention_mask", [1] * len(full_ids))
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

        # Truncate from left; keep end (assistant completion)
        if len(full_ids) > max_length:
            full_ids = full_ids[-max_length:]
            attn = attn[-max_length:]
            labels = labels[-max_length:]

        if all(x == -100 for x in labels):
            return None
        return {"input_ids": full_ids, "attention_mask": attn, "labels": labels}

    for ex in examples:
        # Multi-turn legacy input/output -> convert to history + current user
        if "input" in ex and "output" in ex:
            turns = _parse_legacy_multiturn_input(str(ex["input"]))
            if not turns:
                continue
            history: List[Dict[str, str]] = []
            current_user = None
            for t in turns:
                if t["assistant"] is None:
                    current_user = t["user"]
                    break
                history.append({"user": t["user"], "assistant": t["assistant"]})
            if not current_user:
                continue

            prompt_text = format_chat_prompt_llama2(
                user_message=current_user,
                system_prompt=SYSTEM_PROMPT,
                conversation_history=history if history else None,
            )
            full_text = format_for_training_llama2(
                question=current_user,
                answer=str(ex["output"]),
                system_prompt=SYSTEM_PROMPT,
                conversation_history=history if history else None,
            )
            built = tokenize_pair(prompt_text, full_text)
            if built is not None:
                rows.append(built)
            continue

        # Single-turn question/answer
        if "question" in ex and "answer" in ex:
            q = str(ex["question"]).strip()
            a = str(ex["answer"]).strip()
            if not q or not a:
                continue
            prompt_text = format_chat_prompt_llama2(user_message=q, system_prompt=SYSTEM_PROMPT, conversation_history=None)
            full_text = format_for_training_llama2(question=q, answer=a, system_prompt=SYSTEM_PROMPT, conversation_history=None)
            built = tokenize_pair(prompt_text, full_text)
            if built is not None:
                rows.append(built)
            continue

    print(f"Loaded {len(examples)} raw examples; prepared {len(rows)} tokenized examples using chat_format={chat_format}")
    return Dataset.from_list(rows)


def train(model, tokenizer, dataset, output_dir,
          num_train_epochs, per_device_train_batch_size,
          learning_rate, warmup_steps, logging_steps, save_steps,
          gradient_accumulation_steps=16, early_stopping_patience=3, dataloader_num_workers=0,
          data_collator=None):
    """Train the model using supervised fine-tuning with efficiency optimizations."""
    print(f"Training with {len(dataset)} examples")

    # Split dataset for train/validation (90/10)
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']

    print(f"Train set: {len(train_dataset)}, Validation set: {len(eval_dataset)}")

    # Build training arguments
    # Use eval_strategy (newer transformers API) instead of evaluation_strategy
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=save_steps,
        save_strategy="steps",
        eval_strategy="steps",  # Use eval_strategy (newer API) instead of evaluation_strategy
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        bf16=False,
        lr_scheduler_type="cosine",
        no_cuda=not torch.cuda.is_available(),
        dataloader_pin_memory=False,
        dataloader_num_workers=dataloader_num_workers,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="none",
        save_total_limit=3,
        gradient_checkpointing=True,  # Enable gradient checkpointing
    )

    # When using gradient checkpointing with LoRA-only training, ensure inputs require grad.
    # Also ensure use_cache is disabled (transformers will warn otherwise).
    if getattr(training_args, "gradient_checkpointing", False):
        _enable_input_require_grads(model)
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    # Data collator
    if data_collator is None:
        # Default to completion-only masking for legacy <|assistant|> formatted strings
        data_collator = DataCollatorForCompletionOnly(
            tokenizer=tokenizer,
            response_template="<|assistant|>",
            mlm=False,
        )

    # Add early stopping to prevent overfitting
    callbacks = [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    print(f"Starting fine-tuning with early stopping (patience={early_stopping_patience})...")
    trainer.train()

    # Save the best model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    print(f"Model saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fine-Tuning with TinyLlama Chat Format")
    parser.add_argument("--dataset_path", type=str, default="data_output/qa_dataset_cleaned.jsonl")
    parser.add_argument("--output_dir", type=str, default="fine_tuned_weights")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument(
        "--use_chat_template",
        action="store_true",
        help="Use tokenizer.chat_template to build training examples and completion-only labels (required for Llama 3.1)."
    )
    parser.add_argument(
        "--chat_format",
        type=str,
        default=None,
        choices=["auto", "llama2"],
        help="Chat formatting mode for completion-only labels. If set, overrides --use_chat_template. "
             "Use 'auto' for Llama 3.1; use 'llama2' for Llama 2 chat models."
    )
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                        help="Number of evaluations with no improvement before stopping")
    parser.add_argument("--dataloader_num_workers", type=int, default=0,
                        help="Number of workers for dataloader")
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="Load model in 8-bit precision (requires bitsandbytes)")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Load model in 4-bit precision (requires bitsandbytes)")

    args = parser.parse_args()

    # Check for conflicting quantization arguments
    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("Cannot use both --load_in_8bit and --load_in_4bit at the same time")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    print(f"Quantization: 8bit={args.load_in_8bit}, 4bit={args.load_in_4bit}")
    
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        for_training=not (args.load_in_8bit or args.load_in_4bit),
        enable_gradient_checkpointing=False,  # Trainer enables checkpointing via TrainingArguments
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit
    )
    
    # CRITICAL: Prepare model for quantization training BEFORE LoRA
    # This unfreezes necessary parameters for training
    if args.load_in_8bit or args.load_in_4bit:
        print("Preparing model for quantization training...")
        model = prepare_model_for_kbit_training(model)
        print("? Model prepared for quantization training")
    
    # Configure LoRA - use auto target_modules detection for better compatibility
    print("Configuring LoRA...")
    
    # Try to detect target modules automatically
    try:
        # Get model's module names
        module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                module_names.add(name.split('.')[-1])
        
        # Common transformer module names
        possible_targets = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj", 
                           "query", "value", "key", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        target_modules = [m for m in possible_targets if m in module_names]
        
        if not target_modules:
            # Fallback to common names
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
            print(f"Warning: Using default target_modules: {target_modules}")
        else:
            print(f"Detected target_modules: {target_modules}")
    except Exception as e:
        print(f"Warning: Could not auto-detect modules, using defaults: {e}")
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    lora_config = LoraConfig(
        r=16,  # LoRA rank
        lora_alpha=32,  # LoRA alpha
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA to model (with fallback if module names don't match this architecture)
    print("Applying LoRA to model...")
    try:
        model = get_peft_model(model, lora_config)
    except Exception as e:
        print(f"Warning: LoRA application failed with detected target_modules={target_modules}: {e}")
        fallback_targets = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        print(f"Retrying LoRA with fallback target_modules={fallback_targets} ...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=fallback_targets,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    
    # CRITICAL: After LoRA, verify and enable training
    # LoRA should automatically make its parameters trainable
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    
    print("? LoRA applied successfully")
    
    # Print model info
    total_params = model.num_parameters()
    trainable_params = model.num_parameters(only_trainable=True)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    
    # Verify that model has trainable parameters
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_count == 0:
        raise RuntimeError("ERROR: No trainable parameters found! LoRA setup failed.")
    
    # Verify LoRA layers are trainable
    lora_trainable = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name.lower() and p.requires_grad)
    if lora_trainable == 0:
        raise RuntimeError("ERROR: LoRA layers are not trainable! Check LoRA configuration.")
    
    print(f"Trainable parameters verified: {trainable_count:,}")
    print(f"LoRA trainable parameters: {lora_trainable:,}")
    
    # CRITICAL: Enable training mode
    model.train()
    
    # Verify training mode
    if not model.training:
        raise RuntimeError("ERROR: Model is not in training mode!")
    print("? Model set to training mode")
    
    # Note: Gradient checkpointing is enabled via TrainingArguments (gradient_checkpointing=True)
    # Trainer will handle it automatically, no need to enable manually here
    
    # Final verification: Check trainable parameters
    sample_param = next((p for p in model.parameters() if p.requires_grad), None)
    if sample_param is None:
        raise RuntimeError("ERROR: No trainable parameters found after final check!")
    
    print(f"? Sample trainable parameter: shape={sample_param.shape}, requires_grad={sample_param.requires_grad}")
    
    # Verify model is on correct device
    device_info = f"Device: {next(model.parameters()).device}"
    print(f"? {device_info}")
    
    print(f"? Model ready for training!")
    
    chat_mode = args.chat_format
    if chat_mode is None:
        chat_mode = "auto" if args.use_chat_template else None

    if chat_mode is not None:
        dataset = load_dataset_with_chat_format(args.dataset_path, tokenizer, max_length=args.max_seq_length, chat_format=chat_mode)
        collator = DataCollatorForCausalLMWithLabels(tokenizer=tokenizer)
    else:
        dataset = load_dataset(args.dataset_path, tokenizer, max_length=args.max_seq_length)
        collator = DataCollatorForCompletionOnly(tokenizer=tokenizer, response_template="<|assistant|>", mlm=False)

    train(
        model, tokenizer, dataset, args.output_dir,
        args.num_train_epochs, args.per_device_train_batch_size,
        args.learning_rate, args.warmup_steps,
        args.logging_steps, args.save_steps,
        args.gradient_accumulation_steps,
        args.early_stopping_patience,
        args.dataloader_num_workers,
        data_collator=collator,
    )

    print("Fine-tuning completed successfully!")


if __name__ == "__main__":
    main()

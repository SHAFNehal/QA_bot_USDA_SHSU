"""
Supervised Fine-Tuning (SFT) Script for LLM QA Models

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
from src.utils.model_utils import load_model_and_tokenizer, SYSTEM_PROMPT, format_for_training


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

        # Find the response template tokens
        response_token_ids = self.tokenizer.encode(
            self.response_template,
            add_special_tokens=False
        )

        # Mask everything before the response template with -100
        for i in range(len(labels)):
            input_ids = batch["input_ids"][i].tolist()

            # Find the position of response template
            response_start = None
            for j in range(len(input_ids) - len(response_token_ids) + 1):
                if input_ids[j:j + len(response_token_ids)] == response_token_ids:
                    response_start = j + len(response_token_ids)
                    break

            # If found, mask everything before it
            if response_start is not None:
                labels[i, :response_start] = -100
            else:
                # If not found, mask the entire sequence (shouldn't happen)
                labels[i, :] = -100

            # Also mask padding tokens
            if self.tokenizer.pad_token_id is not None:
                labels[i][labels[i] == self.tokenizer.pad_token_id] = -100

        batch["labels"] = labels
        return batch


def load_dataset(dataset_path, tokenizer, max_length=2048):
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
            padding="max_length",
            max_length=max_length,
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    return tokenized_dataset


def train(model, tokenizer, dataset, output_dir, num_train_epochs,
          per_device_train_batch_size, learning_rate, warmup_steps,
          logging_steps, save_steps, gradient_accumulation_steps=4,
          early_stopping_patience=3):
    """Train the full model using supervised fine-tuning with efficiency optimizations."""
    print(f"Training with {len(dataset)} examples")

    # Split dataset for train/validation (90/10)
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']

    print(f"Train set: {len(train_dataset)}, Validation set: {len(eval_dataset)}")

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
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        lr_scheduler_type="cosine",
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=None,
        save_total_limit=3,
    )

    # Use custom collator for completion-only training (loss only on response)
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

    print(f"Starting training with early stopping (patience={early_stopping_patience})...")
    trainer.train()

    # Save the best model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="SFT with TinyLlama Chat Format")
    parser.add_argument("--dataset_path", type=str, default="data_output/qa_dataset_cleaned.jsonl")
    parser.add_argument("--output_dir", type=str, default="fine_tuned_sft_weights")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                        help="Number of evaluations with no improvement before stopping")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        for_training=True,
        enable_gradient_checkpointing=True
    )
    dataset = load_dataset(args.dataset_path, tokenizer, max_length=args.max_seq_length)

    train(
        model, tokenizer, dataset, args.output_dir,
        args.num_train_epochs, args.per_device_train_batch_size,
        args.learning_rate, args.warmup_steps,
        args.logging_steps, args.save_steps,
        args.gradient_accumulation_steps,
        args.early_stopping_patience
    )

    print("Fine-tuning (SFT) completed successfully!")


if __name__ == "__main__":
    main()

"""
Simple LoRA Fine-tuning Script for LLM QA Models
"""

import os
import json
import argparse
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


def load_model_and_tokenizer(model_name):
    """Load model and tokenizer with LoRA."""
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Apply LoRA
    print("Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


def load_dataset(dataset_path, tokenizer):
    """Load and format QA dataset."""
    print(f"Loading dataset from: {dataset_path}")
    
    # Load QA pairs
    qa_pairs = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                if 'question' in item and 'answer' in item:
                    qa_pairs.append(item)
            except json.JSONDecodeError:
                continue
    
    print(f"Loaded {len(qa_pairs)} QA pairs")
    
    # Format for training
    formatted_texts = []
    for qa_pair in qa_pairs:
        question = qa_pair['question'].strip()
        answer = qa_pair['answer'].strip()
        formatted_text = f"Question: {question}\nAnswer: {answer}"
        formatted_texts.append(formatted_text)
    
    # Create dataset
    dataset = Dataset.from_dict({"text": formatted_texts})
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors=None
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset


def train(model, tokenizer, dataset, output_dir, num_train_epochs, per_device_train_batch_size, learning_rate, warmup_steps, logging_steps, save_steps):
    """Train the model."""
    print(f"Training with {len(dataset)} examples")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_strategy="steps",
        fp16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=None,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Simple LoRA Fine-tuning")
    parser.add_argument("--dataset_path", type=str, default="data_output/qa_dataset_cleaned.jsonl")
    parser.add_argument("--output_dir", type=str, default="fine_tuned_weights")
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    
    # Load dataset
    dataset = load_dataset(args.dataset_path, tokenizer)
    
    # Train model
    train(
        model, tokenizer, dataset, args.output_dir,
        args.num_train_epochs, args.per_device_train_batch_size, args.learning_rate,
        args.warmup_steps, args.logging_steps, args.save_steps
    )
    
    print("Fine-tuning completed successfully!")


if __name__ == "__main__":
    main()
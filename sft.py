"""
Supervised Fine-Tuning (SFT) Script for LLM QA Models
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


def load_model_and_tokenizer(model_name):
    """Load base model and tokenizer """
    print(f"Loading base model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    return model, tokenizer


def load_dataset(dataset_path, tokenizer):
    """Load and format QA dataset for training"""
    print(f"Loading dataset from: {dataset_path}")
    
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
    
    formatted_texts = []
    for qa_pair in qa_pairs:
        question = qa_pair['question'].strip()
        answer = qa_pair['answer'].strip()
        formatted_text = f"Question: {question}\nAnswer: {answer}"
        formatted_texts.append(formatted_text)
    
    dataset = Dataset.from_dict({"text": formatted_texts})
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset


def train(model, tokenizer, dataset, output_dir, num_train_epochs,
          per_device_train_batch_size, learning_rate, warmup_steps,
          logging_steps, save_steps):
    """Train the full model using supervised fine-tuning"""
    print(f"Training with {len(dataset)} examples")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_strategy="steps",
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=None,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="SFT without LoRA")
    parser.add_argument("--dataset_path", type=str, default="data_output/qa_dataset_cleaned.jsonl")
    parser.add_argument("--output_dir", type=str, default="fine_tuned_sft_weights")
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args.model_name)
    dataset = load_dataset(args.dataset_path, tokenizer)

    train(
        model, tokenizer, dataset, args.output_dir,
        args.num_train_epochs, args.per_device_train_batch_size,
        args.learning_rate, args.warmup_steps,
        args.logging_steps, args.save_steps
    )

    print("Fine-tuning (SFT) completed successfully!")


if __name__ == "__main__":
    main()

"""
Inference Script for Fine-tuned QA Model

This script loads a fine-tuned model and performs inference on questions.
Uses the same TinyLlama chat format as training for consistency.
"""

import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
from typing import List, Dict, Optional
from src.utils.model_utils import SYSTEM_PROMPT, format_chat_prompt


class QAInference:
    def __init__(self, base_model_name: str, peft_model_path: str = None,
                 device: str = "auto", max_history_turns: int = 5):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_history_turns = max_history_turns
        self.conversation_history: List[Dict[str, str]] = []

        print(f"Loading base model: {base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,  # Changed from torch_dtype to dtype (deprecated)
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )

        # Load LoRA/PEFT weights if provided
        if peft_model_path:
            print(f"Loading fine-tuned weights from: {peft_model_path}")
            try:
                self.model = PeftModel.from_pretrained(self.model, peft_model_path)
            except Exception as e:
                print(f"Note: Could not load as PEFT model ({e}), trying as full model...")
                # If not a PEFT model, the weights are already in the base model path

        if self.device == "cpu":
            self.model = self.model.to("cpu")

        self.model.eval()
        print("Model loaded successfully!")

    def format_question(self, question: str) -> str:
        """Format single question for TinyLlama chat format (no history)."""
        return f"<|system|>\n{SYSTEM_PROMPT}</s>\n<|user|>\n{question}</s>\n<|assistant|>\n"

    def format_conversation(self, current_question: str) -> str:
        """Format conversation with history for multi-turn support."""
        formatted = f"<|system|>\n{SYSTEM_PROMPT}</s>\n"

        # Add conversation history (limited to max_history_turns)
        for turn in self.conversation_history[-self.max_history_turns:]:
            formatted += f"<|user|>\n{turn['user']}</s>\n"
            formatted += f"<|assistant|>\n{turn['assistant']}</s>\n"

        # Add current question
        formatted += f"<|user|>\n{current_question}</s>\n<|assistant|>\n"

        return formatted
    
    def generate_answer(self, question: str, max_new_tokens: int = 256,
                        temperature: float = 0.7, top_p: float = 0.9,
                        use_history: bool = True) -> str:
        """Generate answer for a given question, optionally using conversation history."""
        # Format with or without history
        if use_history and self.conversation_history:
            formatted_input = self.format_conversation(question)
        else:
            formatted_input = self.format_question(question)

        # Tokenize
        inputs = self.tokenizer(
            formatted_input,
            return_tensors="pt",
            truncation=True,
            max_length=2048  # Increased for conversation history
        )

        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode and extract answer
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Find the assistant response after the last <|assistant|> tag
        if '<|assistant|>' in response:
            answer = response.split('<|assistant|>')[-1].strip()
        else:
            # Fallback to original method
            answer = response[len(formatted_input):].strip()

        # Clean up the answer
        if '</s>' in answer:
            answer = answer.split('</s>')[0].strip()

        # Update conversation history
        if use_history:
            self.conversation_history.append({
                'user': question,
                'assistant': answer
            })

        return answer

    def clear_history(self):
        """Clear conversation history to start a new conversation."""
        self.conversation_history = []
        print("Conversation history cleared.")

    def interactive_mode(self):
        """Run interactive Q&A session with conversation memory."""
        print("=" * 60)
        print("Conversational QA Bot")
        print("Commands: 'quit' to exit, 'clear' to reset conversation")
        print("-" * 60)

        while True:
            question = input("\nYou: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if question.lower() == 'clear':
                self.clear_history()
                continue

            if not question:
                continue

            print("Thinking...")
            answer = self.generate_answer(question)
            print(f"Bot: {answer}")
    
    def batch_inference(self, questions_file: str, output_file: str):
        """Perform batch inference on questions from file."""
        print(f"Loading questions from: {questions_file}")
        
        # Load questions
        with open(questions_file, 'r', encoding='utf-8') as f:
            if questions_file.endswith('.jsonl'):
                questions = []
                for line in f:
                    data = json.loads(line.strip())
                    questions.append(data.get('question', ''))
            else:
                questions = [line.strip() for line in f if line.strip()]
        
        print(f"Processing {len(questions)} questions...")
        
        # Generate answers
        results = []
        for i, question in enumerate(questions):
            print(f"Processing question {i+1}/{len(questions)}")
            answer = self.generate_answer(question)
            results.append({
                'question': question,
                'answer': answer
            })
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Inference with fine-tuned QA model")

    parser.add_argument(
        "--base_model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Base model name"
    )

    parser.add_argument(
        "--peft_model",
        type=str,
        default="./fine_tuned_weights",
        help="Path to fine-tuned weights (LoRA or full model)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use"
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode with conversation memory"
    )

    parser.add_argument(
        "--max_history_turns",
        type=int,
        default=5,
        help="Maximum conversation turns to keep in history"
    )

    parser.add_argument(
        "--questions_file",
        type=str,
        help="File containing questions for batch inference"
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="inference_results.jsonl",
        help="Output file for batch inference results"
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Temperature for generation"
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p for sampling"
    )

    args = parser.parse_args()

    # Initialize inference
    qa_inference = QAInference(
        base_model_name=args.base_model,
        peft_model_path=args.peft_model,
        device=args.device,
        max_history_turns=args.max_history_turns
    )

    if args.interactive:
        qa_inference.interactive_mode()
    elif args.questions_file:
        qa_inference.batch_inference(args.questions_file, args.output_file)
    else:
        # Single question example
        question = "What is machine learning?"
        print(f"Question: {question}")
        answer = qa_inference.generate_answer(
            question,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
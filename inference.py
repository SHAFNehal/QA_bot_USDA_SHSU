"""
Inference Script for Fine-tuned QA Model

This script loads a fine-tuned model and performs inference on questions.
"""

import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json


class QAInference:
    def __init__(self, base_model_name: str, peft_model_path: str, device: str = "auto"):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading base model: {base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        
        # Load LoRA weights
        print(f"Loading LoRA weights from: {peft_model_path}")
        self.model = PeftModel.from_pretrained(self.model, peft_model_path)
        
        if self.device == "cpu":
            self.model = self.model.to("cpu")
        
        self.model.eval()
        print("Model loaded successfully!")
    
    def format_question(self, question: str) -> str:
        """Format question for TinyLlama chat format."""
        return f"<|system|>\nYou are a helpful assistant that answers questions accurately and concisely.</s>\n<|user|>\n{question}</s>\n<|assistant|>\n"
    
    def generate_answer(self, question: str, max_new_tokens: int = 256, 
                       temperature: float = 0.7, top_p: float = 0.9) -> str:
        """Generate answer for a given question."""
        # Format the question
        formatted_input = self.format_question(question)
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_input,
            return_tensors="pt",
            truncation=True,
            max_length=512
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
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response[len(formatted_input):].strip()
        
        # Clean up the answer
        if '</s>' in answer:
            answer = answer.split('</s>')[0].strip()
        
        return answer
    
    def interactive_mode(self):
        """Run interactive Q&A session."""
        print("="*50)
        print("Type your questions (or 'quit' to exit)")
        print("-"*50)
        
        while True:
            question = input("\nQuestion: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            print("Generating answer...")
            answer = self.generate_answer(question)
            print(f"Answer: {answer}")
    
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
        help="Path to LoRA weights"
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
        help="Run in interactive mode"
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
        device=args.device
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
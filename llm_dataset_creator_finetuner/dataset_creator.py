"""
LLM-Based QA Dataset Creator

This script processes text files (.txt, .md, .docx) and generates high-quality
question-answer pairs using a small LLM. The output is saved in JSONL format
for later use in fine-tuning.
"""

import os
import json
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
import torch

from utils.file_processor import process_files
from utils.llm_utils import (
    load_model_and_tokenizer, 
    generate_qa_pairs, 
    create_pipeline,
    generate_qa_with_pipeline,
    get_available_models
)


class QADatasetCreator:
    def __init__(self, model_name: str, device: str = "auto", use_pipeline: bool = True):
        """Initialize the QA dataset creator."""
        self.model_name = model_name
        self.device = device
        self.use_pipeline = use_pipeline
        
        # Load model and tokenizer
        if use_pipeline:
            self.pipe = create_pipeline(model_name, device)
            self.model = None
            self.tokenizer = None
        else:
            self.model, self.tokenizer = load_model_and_tokenizer(model_name, device)
            self.pipe = None
    
    def generate_qa_for_chunk(self, chunk: Dict[str, Any], num_questions: int = 3) -> List[Dict[str, str]]:
        """Generate QA pairs for a single text chunk."""
        try:
            if self.use_pipeline and self.pipe:
                qa_pairs = generate_qa_with_pipeline(
                    self.pipe, 
                    chunk['content'], 
                    num_questions=num_questions
                )
            else:
                qa_pairs = generate_qa_pairs(
                    self.model, 
                    self.tokenizer, 
                    chunk['content'], 
                    num_questions=num_questions
                )
            
            # Add metadata to each QA pair
            for qa_pair in qa_pairs:
                qa_pair.update({
                    'source_file': chunk['source_file'],
                    'file_path': chunk['file_path'],
                    'chunk_index': chunk['chunk_index'],
                    'total_chunks': chunk['total_chunks']
                })
            
            return qa_pairs
            
        except Exception as e:
            print(f"Error generating QA for chunk {chunk['chunk_index']} from {chunk['source_file']}: {e}")
            return []
    
    def create_dataset(self, input_dir: str, output_file: str, 
                      chunk_size: int = 1000, overlap: int = 200,
                      num_questions_per_chunk: int = 3) -> None:
        """Create QA dataset from input files."""
        print(f"Processing files from: {input_dir}")
        print(f"Output will be saved to: {output_file}")
        
        # Process all files
        chunks = process_files(input_dir, chunk_size, overlap)
        print(f"Found {len(chunks)} text chunks to process")
        
        if not chunks:
            print("No text chunks found. Please check your input directory.")
            return
        
        # Generate QA pairs for each chunk
        all_qa_pairs = []
        
        for chunk in tqdm(chunks, desc="Generating QA pairs"):
            qa_pairs = self.generate_qa_for_chunk(chunk, num_questions_per_chunk)
            all_qa_pairs.extend(qa_pairs)
        
        print(f"Generated {len(all_qa_pairs)} QA pairs")
        
        # Save to JSONL file
        self.save_to_jsonl(all_qa_pairs, output_file)
        
        # Print statistics
        self.print_statistics(all_qa_pairs)
    
    def save_to_jsonl(self, qa_pairs: List[Dict[str, str]], output_file: str) -> None:
        """Save QA pairs to JSONL file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for qa_pair in qa_pairs:
                f.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')
        
        print(f"Dataset saved to: {output_file}")
    
    def print_statistics(self, qa_pairs: List[Dict[str, str]]) -> None:
        """Print dataset statistics."""
        if not qa_pairs:
            return
        
        # Count unique source files
        source_files = set(qa_pair['source_file'] for qa_pair in qa_pairs)
        
        # Calculate average question and answer lengths
        question_lengths = [len(qa_pair['question']) for qa_pair in qa_pairs]
        answer_lengths = [len(qa_pair['answer']) for qa_pair in qa_pairs]
        
        avg_question_length = sum(question_lengths) / len(question_lengths)
        avg_answer_length = sum(answer_lengths) / len(answer_lengths)
        
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        print(f"Total QA pairs: {len(qa_pairs)}")
        print(f"Source files: {len(source_files)}")
        print(f"Average question length: {avg_question_length:.1f} characters")
        print(f"Average answer length: {avg_answer_length:.1f} characters")
        print(f"Source files: {', '.join(sorted(source_files))}")
        print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Create QA dataset from text files using LLM")
    
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="data_input",
        help="Directory containing input files (.txt, .md, .docx)"
    )
    
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="data_output/qa_dataset.jsonl",
        help="Output file path for the QA dataset"
    )
    
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", # << This is where you can change the model
        help="Hugging Face model name for QA generation"
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda", # << This is where you can change the device
        choices=["auto", "cpu", "cuda"],
        help="Device to use for model inference"
    )
    
    parser.add_argument(
        "--use_pipeline", 
        action="store_true",
        help="Use transformers pipeline instead of raw model"
    )
    
    parser.add_argument(
        "--chunk_size", 
        type=int, 
        default=1000,
        help="Size of text chunks to process"
    )
    
    parser.add_argument(
        "--overlap", 
        type=int, 
        default=200,
        help="Overlap between text chunks"
    )
    
    parser.add_argument(
        "--num_questions", 
        type=int, 
        default=3,
        help="Number of questions to generate per text chunk"
    )
    
    parser.add_argument(
        "--list_models", 
        action="store_true",
        help="List available models and exit"
    )
    
    args = parser.parse_args()
    
    if args.list_models:
        print("Available models:")
        for model in get_available_models():
            print(f"  - {model}")
        return
    
    # Create QA dataset creator
    creator = QADatasetCreator(
        model_name=args.model_name,
        device=args.device,
        use_pipeline=args.use_pipeline
    )
    
    # Create dataset
    creator.create_dataset(
        input_dir=args.input_dir,
        output_file=args.output_file,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        num_questions_per_chunk=args.num_questions
    )


if __name__ == "__main__":
    main() 
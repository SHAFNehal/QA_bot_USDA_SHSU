"""
LLM-Based QA Dataset Creator

This script processes text files (.txt, .md, .docx) and generates high-quality
question-answer pairs using a small LLM. The output is saved in JSONL format
for later use in fine-tuning.

Features:
- Basic mode: Simple QA generation
- Advanced mode: Diverse QA prompts, semantic filtering, deduplication
- Parallel processing with ThreadPoolExecutor
- Optional paraphrase augmentation (rule-based or LLM-based)
"""

import os
import json
import argparse
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch

from src.utils.file_processor import process_files
from src.utils.llm_utils import (
    load_model_and_tokenizer,
    generate_qa_pairs,
    create_pipeline,
    generate_qa_with_pipeline,
    get_available_models,
    augment_qa_with_llm_paraphrases,
)
from src.dataset.generators.paraphrase_generator import augment_qa_dataset as augment_qa_rule_based
from config import (
    DEFAULT_DATASET_CONFIG,
    PARAPHRASE_CONFIG,
    get_dataset_config,
)


class QADatasetCreator:
    """QA Dataset Creator with basic and advanced modes."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        use_pipeline: bool = True,
        advanced_mode: bool = False,
        similarity_threshold: float = 0.5,
        dedup_threshold: float = 0.8
    ):
        """
        Initialize the QA dataset creator.

        Args:
            model_name: HuggingFace model name
            device: Device to use ('auto', 'cpu', 'cuda')
            use_pipeline: Use transformers pipeline instead of raw model
            advanced_mode: Enable diverse prompts, semantic filtering, deduplication
            similarity_threshold: Minimum similarity for QA relevance filtering
            dedup_threshold: Threshold for duplicate detection
        """
        self.model_name = model_name
        self.device = device
        self.use_pipeline = use_pipeline
        self.advanced_mode = advanced_mode
        self.similarity_threshold = similarity_threshold
        self.dedup_threshold = dedup_threshold
        self.semantic_model = None

        # Load model/pipeline
        if use_pipeline:
            self.pipe = create_pipeline(model_name, device)
            self.model = None
            self.tokenizer = None
        else:
            self.model, self.tokenizer = load_model_and_tokenizer(model_name, device)
            self.pipe = None

        # Load semantic model for advanced mode
        if advanced_mode:
            try:
                from sentence_transformers import SentenceTransformer
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("Semantic model loaded for advanced filtering")
            except ImportError:
                print("Warning: sentence-transformers not installed. Install with:")
                print("  pip install sentence-transformers")
                print("Continuing without semantic filtering...")
                self.advanced_mode = False

    def generate_diverse_qa_prompts(self, chunk: str) -> List[str]:
        """Generate diverse QA prompts for better coverage."""
        templates = [
            f"Read the following content and generate 2 factual question-answer pairs:\n\n\"\"\"{chunk}\"\"\"",
            f"Generate 2 definition-based question-answer pairs from this content:\n\"\"\"{chunk}\"\"\"",
            f"Create 2 reasoning (why/how) question-answer pairs based on:\n\"\"\"{chunk}\"\"\"",
            f"Provide 2 comparison question-answer pairs from:\n\"\"\"{chunk}\"\"\"",
            f"Write 2 list-based or multi-point answer question-answer pairs from:\n\"\"\"{chunk}\"\"\"",
            f"Provide 2 procedural (how-to) question-answer pairs from this content:\n\"\"\"{chunk}\"\"\""
        ]
        return templates

    def filter_relevant_pairs(
        self,
        qa_pairs: List[Dict[str, str]],
        chunk: str
    ) -> List[Dict[str, str]]:
        """Filter QA pairs by semantic relevance to source chunk."""
        if not self.semantic_model:
            return qa_pairs

        from sentence_transformers import util

        filtered = []
        chunk_embedding = self.semantic_model.encode(chunk, convert_to_tensor=True)

        for qa in qa_pairs:
            question_embedding = self.semantic_model.encode(
                qa['question'],
                convert_to_tensor=True
            )
            score = util.cos_sim(question_embedding, chunk_embedding).item()
            if score >= self.similarity_threshold:
                filtered.append(qa)

        return filtered

    def remove_duplicate_questions(
        self,
        qa_pairs: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Remove semantically similar duplicate questions using batched encoding."""
        if not self.semantic_model or not qa_pairs:
            return qa_pairs

        from sentence_transformers import util
        import numpy as np

        # Batch encode all questions at once (much faster than one at a time)
        questions = [qa['question'] for qa in qa_pairs]
        embeddings = self.semantic_model.encode(questions, convert_to_tensor=True, show_progress_bar=False)

        # Use clustering-based deduplication for efficiency
        unique_indices = []
        seen_embeddings = []

        for i, emb in enumerate(embeddings):
            is_duplicate = False

            if seen_embeddings:
                # Compare against all seen embeddings at once
                seen_tensor = torch.stack(seen_embeddings)
                similarities = util.cos_sim(emb.unsqueeze(0), seen_tensor)[0]
                if torch.any(similarities >= self.dedup_threshold):
                    is_duplicate = True

            if not is_duplicate:
                seen_embeddings.append(emb)
                unique_indices.append(i)

        return [qa_pairs[i] for i in unique_indices]

    def generate_qa_for_chunk(
        self,
        chunk: Dict[str, Any],
        num_questions: int = 3
    ) -> List[Dict[str, str]]:
        """Generate QA pairs for a single text chunk."""
        try:
            all_qa = []

            if self.advanced_mode:
                # Advanced mode: use diverse prompts
                prompts = self.generate_diverse_qa_prompts(chunk['content'])
                for prompt in prompts:
                    pairs = generate_qa_with_pipeline(self.pipe, prompt)
                    for qa_pair in pairs:
                        qa_pair.update({
                            'source_file': chunk['source_file'],
                            'file_path': chunk['file_path'],
                            'chunk_index': chunk['chunk_index'],
                            'total_chunks': chunk['total_chunks']
                        })
                    all_qa.extend(pairs)

                # Filter by relevance
                all_qa = self.filter_relevant_pairs(all_qa, chunk['content'])
                # Remove duplicates
                all_qa = self.remove_duplicate_questions(all_qa)
            else:
                # Basic mode
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

                all_qa = qa_pairs

            return all_qa

        except Exception as e:
            print(f"Error generating QA for chunk {chunk['chunk_index']} from {chunk['source_file']}: {e}")
            return []

    def create_dataset(
        self,
        input_dir: str,
        output_file: str,
        chunk_size: int = 1000,
        overlap: int = 200,
        num_questions_per_chunk: int = 3,
        num_workers: int = 1,
        augment_paraphrases: int = 0,
        use_llm_paraphrases: bool = False
    ) -> None:
        """Create QA dataset from input files.

        Args:
            input_dir: Directory containing input files
            output_file: Output JSONL file path
            chunk_size: Size of text chunks
            overlap: Overlap between chunks
            num_questions_per_chunk: Number of QA pairs per chunk
            num_workers: Number of parallel workers
            augment_paraphrases: Number of paraphrase variations per question (0 to disable)
            use_llm_paraphrases: Use LLM for paraphrase generation instead of rules
        """
        print(f"Processing files from: {input_dir}")
        print(f"Output will be saved to: {output_file}")
        print(f"Mode: {'Advanced' if self.advanced_mode else 'Basic'}")
        if augment_paraphrases > 0:
            print(f"Paraphrase augmentation: {augment_paraphrases} variations ({'LLM-based' if use_llm_paraphrases else 'rule-based'})")

        # Process all files
        chunks = process_files(input_dir, chunk_size, overlap)
        print(f"Found {len(chunks)} text chunks to process")

        if not chunks:
            print("No text chunks found. Please check your input directory.")
            return

        # Generate QA pairs
        all_qa_pairs = []

        if num_workers > 1 and self.use_pipeline:
            # Parallel processing
            print(f"Using {num_workers} workers for parallel processing")
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                results = list(tqdm(
                    executor.map(
                        lambda c: self.generate_qa_for_chunk(c, num_questions_per_chunk),
                        chunks
                    ),
                    total=len(chunks),
                    desc="Generating QA pairs"
                ))
            for qa_list in results:
                all_qa_pairs.extend(qa_list)
        else:
            # Sequential processing
            for chunk in tqdm(chunks, desc="Generating QA pairs"):
                qa_pairs = self.generate_qa_for_chunk(chunk, num_questions_per_chunk)
                all_qa_pairs.extend(qa_pairs)

        print(f"Generated {len(all_qa_pairs)} QA pairs")

        # Paraphrase augmentation (before deduplication)
        if augment_paraphrases > 0 and all_qa_pairs:
            original_count = len(all_qa_pairs)
            print(f"Augmenting with {augment_paraphrases} paraphrases per question...")

            if use_llm_paraphrases and self.pipe:
                # LLM-based paraphrase generation
                all_qa_pairs = augment_qa_with_llm_paraphrases(
                    self.pipe,
                    all_qa_pairs,
                    variations_per_question=augment_paraphrases
                )
            else:
                # Rule-based paraphrase generation
                all_qa_pairs = augment_qa_rule_based(
                    all_qa_pairs,
                    variations_per_question=augment_paraphrases
                )

            print(f"Dataset size after augmentation: {len(all_qa_pairs)} (was {original_count})")

        # Final deduplication across all chunks
        if self.advanced_mode and self.semantic_model:
            original_count = len(all_qa_pairs)
            all_qa_pairs = self.remove_duplicate_questions(all_qa_pairs)
            print(f"Removed {original_count - len(all_qa_pairs)} duplicates across chunks")

        # Save to JSONL file
        self.save_to_jsonl(all_qa_pairs, output_file)

        # Print statistics
        self.print_statistics(all_qa_pairs)

    def save_to_jsonl(self, qa_pairs: List[Dict[str, str]], output_file: str) -> None:
        """Save QA pairs to JSONL file."""
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            for qa_pair in qa_pairs:
                f.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')

        print(f"Dataset saved to: {output_file}")

    def print_statistics(self, qa_pairs: List[Dict[str, str]]) -> None:
        """Print dataset statistics."""
        if not qa_pairs:
            print("No QA pairs generated.")
            return

        # Count unique source files
        source_files = set(qa_pair.get('source_file', 'unknown') for qa_pair in qa_pairs)

        # Calculate average question and answer lengths
        question_lengths = [len(qa_pair['question']) for qa_pair in qa_pairs]
        answer_lengths = [len(qa_pair['answer']) for qa_pair in qa_pairs]

        avg_question_length = sum(question_lengths) / len(question_lengths)
        avg_answer_length = sum(answer_lengths) / len(answer_lengths)

        print("\n" + "=" * 50)
        print("DATASET STATISTICS")
        print("=" * 50)
        print(f"Total QA pairs: {len(qa_pairs)}")
        print(f"Source files: {len(source_files)}")
        print(f"Average question length: {avg_question_length:.1f} characters")
        print(f"Average answer length: {avg_answer_length:.1f} characters")
        if len(source_files) <= 10:
            print(f"Files: {', '.join(sorted(source_files))}")
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Create QA dataset from text files using LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic mode
  python dataset_creator.py --input_dir data_input --output_file data_output/qa.jsonl

  # Advanced mode with semantic filtering
  python dataset_creator.py --input_dir data_input --output_file data_output/qa.jsonl --advanced

  # Parallel processing
  python dataset_creator.py --input_dir data_input --output_file data_output/qa.jsonl --workers 4
        """
    )

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
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Hugging Face model name for QA generation"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for model inference"
    )

    parser.add_argument(
        "--use_pipeline",
        action="store_true",
        default=True,
        help="Use transformers pipeline instead of raw model"
    )

    parser.add_argument(
        "--no_pipeline",
        action="store_true",
        help="Use raw model instead of pipeline"
    )

    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Enable advanced mode with diverse prompts, semantic filtering, and deduplication"
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=DEFAULT_DATASET_CONFIG.get("chunk_size", 1000),
        help="Size of text chunks to process"
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_DATASET_CONFIG.get("overlap", 200),
        help="Overlap between text chunks"
    )

    parser.add_argument(
        "--num_questions",
        type=int,
        default=DEFAULT_DATASET_CONFIG.get("num_questions_per_chunk", 3),
        help="Number of questions to generate per text chunk (basic mode)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for processing"
    )

    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.5,
        help="Minimum similarity for QA relevance filtering (advanced mode)"
    )

    parser.add_argument(
        "--dedup_threshold",
        type=float,
        default=0.8,
        help="Threshold for duplicate detection (advanced mode)"
    )

    parser.add_argument(
        "--list_models",
        action="store_true",
        help="List available models and exit"
    )

    parser.add_argument(
        "--augment_paraphrases",
        type=int,
        default=PARAPHRASE_CONFIG.get("variations_per_question", 0),
        help="Number of paraphrase variations per question (0 to disable)"
    )

    parser.add_argument(
        "--use_llm_paraphrases",
        action="store_true",
        help="Use LLM for paraphrase generation instead of rule-based"
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
        use_pipeline=not args.no_pipeline,
        advanced_mode=args.advanced,
        similarity_threshold=args.similarity_threshold,
        dedup_threshold=args.dedup_threshold
    )

    # Create dataset
    creator.create_dataset(
        input_dir=args.input_dir,
        output_file=args.output_file,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        num_questions_per_chunk=args.num_questions,
        num_workers=args.workers,
        augment_paraphrases=args.augment_paraphrases,
        use_llm_paraphrases=args.use_llm_paraphrases
    )


if __name__ == "__main__":
    main()

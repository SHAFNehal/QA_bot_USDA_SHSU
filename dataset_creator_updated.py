import os
import json
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer, util

from utils.file_processor import process_files  # Assumed available
from utils.llm_utils import (
    create_pipeline,
    generate_qa_with_pipeline,
    get_available_models
)


class AdvancedQADatasetCreator:
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self.pipe = create_pipeline(model_name, device)
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

    def generate_diverse_qa_prompts(self, chunk: str) -> List[str]:
        templates = [
            f"Read the following content and generate 2 factual question-answer pairs:\n\n\"\"\"{chunk}\"\"\"",
            f"Generate 2 definition-based question-answer pairs from this content:\n\"\"\"{chunk}\"\"\"",
            f"Create 2 reasoning (why/how) question-answer pairs based on:\n\"\"\"{chunk}\"\"\"",
            f"Provide 2 comparison question-answer pairs from:\n\"\"\"{chunk}\"\"\"",
            f"Write 2 list-based or multi-point answer question-answer pairs from:\n\"\"\"{chunk}\"\"\"",
            f"Provide 2 procedural (how-to) question-answer pairs from this content:\n\"\"\"{chunk}\"\"\""
        ]
        return templates

    def filter_relevant_pairs(self, qa_pairs: List[Dict[str, str]], chunk: str, threshold: float = 0.5) -> List[Dict[str, str]]:
        filtered = []
        for qa in qa_pairs:
            score = util.cos_sim(
                self.semantic_model.encode(qa['question'], convert_to_tensor=True),
                self.semantic_model.encode(chunk, convert_to_tensor=True)
            ).item()
            if score >= threshold:
                filtered.append(qa)
        return filtered

    def remove_duplicate_questions(self, qa_pairs: List[Dict[str, str]], threshold: float = 0.8) -> List[Dict[str, str]]:
        unique_pairs = []
        seen_embeddings = []
        for qa in qa_pairs:
            emb = self.semantic_model.encode(qa['question'], convert_to_tensor=True)
            if all(util.cos_sim(emb, seen)[0][0].item() < threshold for seen in seen_embeddings):
                seen_embeddings.append(emb)
                unique_pairs.append(qa)
        return unique_pairs

    def generate_qa_for_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, str]]:
        try:
            all_qa = []
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
                filtered = self.filter_relevant_pairs(pairs, chunk['content'])
                all_qa.extend(filtered)
            deduplicated_qa = self.remove_duplicate_questions(all_qa)
            return deduplicated_qa
        except Exception as e:
            print(f"Error generating QA for chunk {chunk['chunk_index']} from {chunk['source_file']}: {e}")
            return []

    def create_dataset(self, input_dir: str, output_file: str,
                       chunk_size: int = 1000, overlap: int = 200) -> None:
        print(f"Processing files from: {input_dir}")
        chunks = process_files(input_dir, chunk_size, overlap)
        print(f"Found {len(chunks)} text chunks to process")
        if not chunks:
            print("No text chunks found. Please check your input directory.")
            return

        all_qa_pairs = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(tqdm(executor.map(self.generate_qa_for_chunk, chunks), total=len(chunks)))

        for qa_list in results:
            all_qa_pairs.extend(qa_list)

        self.save_to_jsonl(all_qa_pairs, output_file)
        self.print_statistics(all_qa_pairs)

    def save_to_jsonl(self, qa_pairs: List[Dict[str, str]], output_file: str) -> None:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for qa_pair in qa_pairs:
                f.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')
        print(f"Dataset saved to: {output_file}")

    def print_statistics(self, qa_pairs: List[Dict[str, str]]) -> None:
        if not qa_pairs:
            return
        question_lengths = [len(qa['question']) for qa in qa_pairs]
        answer_lengths = [len(qa['answer']) for qa in qa_pairs]
        print("\n" + "=" * 50)
        print("DATASET STATISTICS")
        print("=" * 50)
        print(f"Total QA pairs: {len(qa_pairs)}")
        print(f"Average question length: {sum(question_lengths)/len(question_lengths):.1f} chars")
        print(f"Average answer length: {sum(answer_lengths)/len(answer_lengths):.1f} chars")
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Advanced QA Dataset Creator with LLM")
    parser.add_argument("--input_dir", type=str, default="data_input")
    parser.add_argument("--output_file", type=str, default="data_output/qa_dataset_advanced.jsonl")
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--list_models", action="store_true")

    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for model in get_available_models():
            print(f"  - {model}")
        return

    creator = AdvancedQADatasetCreator(args.model_name, args.device)
    creator.create_dataset(
        input_dir=args.input_dir,
        output_file=args.output_file,
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )


if __name__ == "__main__":
    main()

"""
Dataset Merger Script

Merges QA datasets with conversational data and multi-turn conversation data
to create a comprehensive training dataset.
"""

import argparse
import os
import random
from typing import List, Dict

from src.utils.file_processor import load_jsonl, save_jsonl
from src.dataset.generators.conversational_data import get_conversational_qa_pairs


def merge_datasets(
    qa_data_path: str = None,
    multiturn_data_path: str = None,
    output_path: str = "data_output/training_dataset.jsonl",
    include_conversational: bool = True,
    conversational_multiplier: int = 3,
    shuffle: bool = True,
    seed: int = 42
) -> List[Dict]:
    """
    Merge multiple data sources into a single training dataset.

    Args:
        qa_data_path: Path to single-turn QA dataset (JSONL)
        multiturn_data_path: Path to multi-turn conversation dataset (JSONL)
        output_path: Path to save merged dataset
        include_conversational: Whether to include built-in conversational pairs
        conversational_multiplier: How many times to repeat conversational data
        shuffle: Whether to shuffle the final dataset
        seed: Random seed for shuffling

    Returns:
        List of merged data items
    """
    all_data = []

    # Load QA data
    if qa_data_path and os.path.isfile(qa_data_path):
        qa_data = load_jsonl(qa_data_path)
        print(f"Loaded {len(qa_data)} QA pairs from {qa_data_path}")
        all_data.extend(qa_data)
    else:
        print(f"No QA data found at {qa_data_path}")

    # Load multi-turn data
    if multiturn_data_path and os.path.isfile(multiturn_data_path):
        multiturn_data = load_jsonl(multiturn_data_path)
        print(f"Loaded {len(multiturn_data)} multi-turn examples from {multiturn_data_path}")
        all_data.extend(multiturn_data)
    else:
        if multiturn_data_path:
            print(f"No multi-turn data found at {multiturn_data_path}")

    # Add conversational data
    if include_conversational:
        conversational_pairs = get_conversational_qa_pairs()
        # Repeat conversational data to ensure adequate representation
        for _ in range(conversational_multiplier):
            all_data.extend(conversational_pairs)
        print(f"Added {len(conversational_pairs) * conversational_multiplier} conversational pairs "
              f"({len(conversational_pairs)} unique x {conversational_multiplier})")

    # Shuffle if requested
    if shuffle:
        random.seed(seed)
        random.shuffle(all_data)
        print(f"Shuffled dataset with seed {seed}")

    # Save merged dataset
    save_jsonl(all_data, output_path)
    print(f"Saved {len(all_data)} total examples to {output_path}")

    # Print statistics
    print_statistics(all_data)

    return all_data


def print_statistics(data: List[Dict]) -> None:
    """Print dataset statistics."""
    print("\n" + "=" * 50)
    print("MERGED DATASET STATISTICS")
    print("=" * 50)

    # Count by type
    type_counts = {}
    for item in data:
        item_type = item.get('type', 'qa')
        type_counts[item_type] = type_counts.get(item_type, 0) + 1

    for item_type, count in sorted(type_counts.items()):
        percentage = (count / len(data)) * 100
        print(f"  {item_type}: {count} ({percentage:.1f}%)")

    # Count format types
    single_turn = sum(1 for item in data if 'question' in item and 'answer' in item)
    multi_turn = sum(1 for item in data if 'input' in item and 'output' in item)

    print("-" * 50)
    print(f"  Single-turn (question/answer): {single_turn}")
    print(f"  Multi-turn (input/output): {multi_turn}")
    print("-" * 50)
    print(f"  TOTAL: {len(data)} examples")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Merge datasets for training")

    parser.add_argument(
        "--qa_data",
        type=str,
        default="data_output/qa_dataset_cleaned.jsonl",
        help="Path to QA dataset"
    )

    parser.add_argument(
        "--multiturn_data",
        type=str,
        default=None,
        help="Path to multi-turn conversation dataset"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data_output/training_dataset.jsonl",
        help="Path to save merged dataset"
    )

    parser.add_argument(
        "--no_conversational",
        action="store_true",
        help="Exclude built-in conversational data"
    )

    parser.add_argument(
        "--conversational_multiplier",
        type=int,
        default=3,
        help="Times to repeat conversational data (for better representation)"
    )

    parser.add_argument(
        "--no_shuffle",
        action="store_true",
        help="Don't shuffle the final dataset"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling"
    )

    args = parser.parse_args()

    merge_datasets(
        qa_data_path=args.qa_data,
        multiturn_data_path=args.multiturn_data,
        output_path=args.output,
        include_conversational=not args.no_conversational,
        conversational_multiplier=args.conversational_multiplier,
        shuffle=not args.no_shuffle,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

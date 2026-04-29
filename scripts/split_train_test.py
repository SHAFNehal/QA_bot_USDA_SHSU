"""
Split training_dataset.jsonl into train (1 - test_ratio) and test (test_ratio).
Train portion overwrites the input file; test portion is written to a separate file.
Test set is never used in training.

Usage (from project root):
  python scripts/split_train_test.py --input data_output/training_dataset.jsonl --test-ratio 0.1 --output-test data_output/test_set.jsonl --seed 42
"""

import argparse
import json
import random


def main():
    parser = argparse.ArgumentParser(description="Split JSONL into train and test; train overwrites input, test to separate file.")
    parser.add_argument("--input", required=True, help="Input JSONL (will be overwritten with train portion)")
    parser.add_argument("--output-test", required=True, help="Output JSONL for test portion (e.g. data_output/test_set.jsonl)")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Fraction for test set (default 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffle")
    args = parser.parse_args()

    if not 0 < args.test_ratio < 1:
        raise ValueError("--test-ratio must be in (0, 1)")

    with open(args.input, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    data = [json.loads(line) for line in lines]
    n = len(data)
    if n == 0:
        raise ValueError(f"No examples in {args.input}")

    rng = random.Random(args.seed)
    rng.shuffle(data)

    n_test = max(1, int(round(n * args.test_ratio)))
    n_train = n - n_test
    train_data = data[:n_train]
    test_data = data[n_train:]

    with open(args.input, "w", encoding="utf-8") as f:
        for obj in train_data:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    with open(args.output_test, "w", encoding="utf-8") as f:
        for obj in test_data:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Split: {n_train} train -> {args.input}, {n_test} test -> {args.output_test} (seed={args.seed})")


if __name__ == "__main__":
    main()

"""
Replicate the 90/10 train/validation split used at training time (seed=42)
and save the validation (test) portion to a JSONL file for evaluation.

Training uses: dataset.train_test_split(test_size=0.1, seed=42)
The validation portion is only used for eval_loss / early stopping and was
never saved. This script reproduces the same split and writes the validation
examples to data_output/validation_split_10pct.jsonl.

Usage (from project root):
  python scripts/save_validation_split.py
  python scripts/save_validation_split.py --dataset data_output/training_dataset.jsonl --output data_output/validation_split_10pct.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

# Same filtering logic as fine_tuner so we keep the same examples (order and count).
def _parse_legacy_multiturn_input(input_text: str) -> List[Dict[str, Optional[str]]]:
    text = input_text or ""
    turns: List[Dict[str, Optional[str]]] = []
    user_tag = "<|user|>"
    asst_tag = "<|assistant|>"
    end_tag = "</s>"
    i = 0
    while True:
        u = text.find(user_tag, i)
        if u == -1:
            break
        u_start = u + len(user_tag)
        u_end = text.find(end_tag, u_start)
        if u_end == -1:
            break
        user_msg = text[u_start:u_end].strip()
        a = text.find(asst_tag, u_end)
        if a == -1:
            turns.append({"user": user_msg, "assistant": None})
            break
        a_start = a + len(asst_tag)
        a_end = text.find(end_tag, a_start)
        if a_end == -1:
            turns.append({"user": user_msg, "assistant": None})
            break
        assistant_msg = text[a_start:a_end].strip()
        if assistant_msg:
            turns.append({"user": user_msg, "assistant": assistant_msg})
            i = a_end + len(end_tag)
            continue
        turns.append({"user": user_msg, "assistant": None})
        break
    return turns


def _keep_example(ex: Dict[str, Any]) -> bool:
    if "question" in ex and "answer" in ex:
        q = ex.get("question")
        a = ex.get("answer")
        if q is None or a is None:
            return False
        q, a = str(q).strip(), str(a).strip()
        return bool(q and a)
    if "input" in ex and "output" in ex:
        raw_in, raw_out = ex.get("input"), ex.get("output")
        if raw_in is None or raw_out is None:
            return False
        out_str = str(raw_out).strip()
        if not out_str:
            return False
        turns = _parse_legacy_multiturn_input(str(raw_in))
        if not turns:
            return False
        for t in turns:
            if t.get("assistant") is None:
                return True
        return False
    return False


def main():
    parser = argparse.ArgumentParser(description="Save validation split (10%%, seed=42) from training dataset")
    parser.add_argument("--dataset", type=str, default=None, help="Path to training_dataset.jsonl")
    parser.add_argument("--output", type=str, default=None, help="Output path for validation JSONL")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (must match training)")
    parser.add_argument("--test-size", type=float, default=0.1, help="Fraction for validation (default 0.1)")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = args.dataset or os.path.join(project_root, "data_output", "training_dataset.jsonl")
    output_path = args.output or os.path.join(project_root, "data_output", "validation_split_10pct.jsonl")

    if not os.path.isfile(dataset_path):
        print(f"Error: Dataset not found: {dataset_path}")
        return 1

    kept: List[Dict[str, Any]] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                if _keep_example(ex):
                    kept.append(ex)
            except json.JSONDecodeError:
                continue

    n = len(kept)
    if n == 0:
        print("Error: No examples passed the filter.")
        return 1

    try:
        from datasets import Dataset
    except ImportError:
        print("Error: 'datasets' required. pip install datasets")
        return 1

    ds = Dataset.from_list(kept)
    split = ds.train_test_split(test_size=args.test_size, seed=args.seed)
    val_ds = split["test"]
    n_val = len(val_ds)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(n_val):
            row = val_ds[i]
            out_val = row.get("output") if "output" in row else row.get("answer")
            if out_val is None or not str(out_val).strip():
                continue
            # Drop null keys so we don't write input: null, output: null on question/answer rows (Dataset adds them)
            row_clean = {k: v for k, v in row.items() if v is not None}
            f.write(json.dumps(row_clean, ensure_ascii=False) + "\n")
            written += 1

    print(f"Dataset: {dataset_path}")
    print(f"Kept (after filter): {n} examples")
    print(f"Validation set (10%): {n_val} examples, written (non-null output/answer): {written}")
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    exit(main())

"""
Clean the test/holdout set for evaluation.

Uses data_cleaner for validation (length, format, duplicates) and adds filtering
for context-dependent questions: questions created for multi-turn (e.g. "How does it work?")
that don't make sense when asked standalone. These are removed so evaluation only
tests standalone questions.

Multi-turn items (input/output format) are kept as-is; evaluate.py replays the full
conversation for them.

Usage (from project root):
  python scripts/clean_test_set.py --input data_output/test_set.jsonl --output data_output/test_set_cleaned.jsonl
  python scripts/clean_test_set.py -i data_output/test_set.jsonl -o data_output/test_set_cleaned.jsonl --relaxed
"""

import argparse
import json
import sys
from pathlib import Path

# Project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.data_cleaner import (
    load_jsonl,
    save_jsonl,
    clean_text,
    clean_qa_dataset,
    is_context_dependent_question,
    is_conversational_input,
    is_valid_question,
    is_valid_answer,
    print_stats,
)
from collections import defaultdict


def extract_user_turns_from_input(raw_input: str) -> list:
    """Extract user turns from multi-turn input string."""
    import re
    pattern = r"<\|user\|>\s*\n(.*?)</s>"
    matches = re.findall(pattern, raw_input, re.DOTALL)
    return [m.strip() for m in matches if m.strip()]


def clean_test_set(
    input_path: str,
    output_path: str,
    strict_mode: bool = True,
    filter_context_dependent: bool = True,
    keep_multiturn: bool = True,
    skip_duplicates: bool = True,
) -> dict:
    """
    Clean test set: validate QA pairs, remove context-dependent standalone questions,
    keep multi-turn items as-is.

    Returns stats dict.
    """
    data = load_jsonl(input_path)
    stats = defaultdict(int)
    stats["total"] = len(data)
    cleaned = []

    for item in data:
        stats["processed"] += 1

        # Multi-turn format: input/output
        if "input" in item and "output" in item:
            if not keep_multiturn:
                stats["skipped_multiturn"] += 1
                continue
            raw_input = item["input"]
            output = str(item["output"]).strip()
            user_turns = extract_user_turns_from_input(raw_input)
            if not user_turns:
                stats["invalid_multiturn"] += 1
                continue
            # Basic output validation
            if len(output) < 5 or len(output) > 1000:
                stats["invalid_multiturn_answer"] += 1
                continue
            cleaned.append(item)
            stats["kept_multiturn"] += 1
            continue

        # Single-turn format: question/answer
        if "question" not in item or "answer" not in item:
            stats["missing_fields"] += 1
            continue

        question = clean_text(str(item["question"]))
        answer = clean_text(str(item["answer"]))

        if not question or not answer:
            stats["empty"] += 1
            continue

        # Filter context-dependent questions (standalone-invalid when asked as single turn)
        if filter_context_dependent and is_context_dependent_question(question):
            stats["context_dependent"] += 1
            continue

        # Conversational: relaxed validation
        is_conv = is_conversational_input(question)
        use_strict = strict_mode and not is_conv

        if not is_valid_question(question, strict_mode=use_strict):
            stats["invalid_question"] += 1
            continue

        if not is_valid_answer(answer, strict_mode=use_strict):
            stats["invalid_answer"] += 1
            continue

        cleaned.append({"question": question, "answer": answer, **{k: v for k, v in item.items() if k not in ("question", "answer")}})
        stats["kept_qa"] += 1

    # Deduplicate if requested (simple hash-based for question/answer)
    if skip_duplicates:
        seen = set()
        deduped = []
        for item in cleaned:
            if "input" in item:
                key = item.get("input", "")[:500] + "|||" + str(item.get("output", ""))[:500]
            else:
                key = item.get("question", "") + "|||" + item.get("answer", "")
            if key in seen:
                stats["duplicate"] += 1
                continue
            seen.add(key)
            deduped.append(item)
        cleaned = deduped

    stats["valid"] = len(cleaned)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(cleaned, output_path)
    return dict(stats)


def main():
    p = argparse.ArgumentParser(description="Clean test/holdout set for evaluation")
    p.add_argument("--input", "-i", required=True, help="Input test set JSONL")
    p.add_argument("--output", "-o", required=True, help="Output cleaned JSONL")
    p.add_argument("--relaxed", action="store_true", help="Relaxed validation (allow shorter inputs)")
    p.add_argument("--keep-context-dependent", action="store_true", help="Do NOT filter context-dependent questions")
    p.add_argument("--keep-duplicates", action="store_true", help="Keep duplicate entries")
    p.add_argument("--verbose", "-v", action="store_true", help="Print detailed stats")
    args = p.parse_args()

    if not Path(args.input).exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    print(f"Loading: {args.input}")
    stats = clean_test_set(
        args.input,
        args.output,
        strict_mode=not args.relaxed,
        filter_context_dependent=not args.keep_context_dependent,
        keep_multiturn=True,
        skip_duplicates=not args.keep_duplicates,
    )

    print(f"\nCleaned test set saved to: {args.output}")
    print(f"Total input: {stats['total']}")
    print(f"Valid output: {stats['valid']}")
    print(f"Removed: {stats['total'] - stats['valid']}")
    if stats.get("context_dependent"):
        print(f"  - Context-dependent (standalone-invalid): {stats['context_dependent']}")
    if stats.get("invalid_question"):
        print(f"  - Invalid question: {stats['invalid_question']}")
    if stats.get("invalid_answer"):
        print(f"  - Invalid answer: {stats['invalid_answer']}")
    if stats.get("kept_multiturn"):
        print(f"  - Multi-turn kept: {stats['kept_multiturn']}")
    if stats.get("kept_qa"):
        print(f"  - Single-turn QA kept: {stats['kept_qa']}")


if __name__ == "__main__":
    main()

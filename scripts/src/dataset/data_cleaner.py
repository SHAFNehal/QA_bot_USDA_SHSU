"""
Enhanced Data Cleaner for QA Dataset

This script cleans QA pairs from JSONL files, with support for both
strict QA validation and relaxed conversational data validation.
"""

import json
import argparse
from typing import List, Dict, Any, Tuple
import re
from collections import defaultdict
from pathlib import Path


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save data to JSONL file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def is_conversational_input(text: str) -> bool:
    """
    Check if the input is a conversational phrase (greeting, thanks, etc.)
    rather than a domain question.
    """
    text_lower = text.lower().strip().rstrip('!?.')

    # Common conversational patterns
    conversational_patterns = [
        # Greetings
        'hi', 'hello', 'hey', 'howdy', 'hiya', 'greetings',
        'good morning', 'good afternoon', 'good evening', 'good night',
        'hi there', 'hello there', "what's up",
        # Gratitude
        'thank', 'thanks', 'appreciate', 'grateful',
        # Farewells
        'bye', 'goodbye', 'see you', 'take care', 'farewell',
        # Acknowledgments
        'okay', 'ok', 'got it', 'i see', 'makes sense', 'understood',
        'alright', 'sure', 'yes', 'no', 'yeah', 'nope',
        # Meta
        'who are you', 'what are you', 'what can you do', 'are you a bot',
        'can you help', 'help me',
        # Clarifications
        "what?", "can you explain", "i don't understand", "clarify",
    ]

    for pattern in conversational_patterns:
        if text_lower == pattern or text_lower.startswith(pattern + ' '):
            return True

    return False


def is_valid_conversational(text: str) -> bool:
    """Validation for conversational inputs (greetings, thanks, etc.)."""
    text = text.strip()

    if not text:
        return False

    # Conversational inputs are typically short
    if len(text) > 100:
        return False

    # Should have at least one character
    if len(text) < 1:
        return False

    return True


def is_valid_question(question: str, strict_mode: bool = True) -> bool:
    """
    Validation for questions.

    Args:
        question: The question text
        strict_mode: If True, apply strict QA validation. If False, allow
                    conversational inputs and command-style queries.
    """
    question = question.strip()

    # Basic checks
    if not question:
        return False

    # Check if it's a conversational input (always allowed if valid)
    if is_conversational_input(question):
        return is_valid_conversational(question)

    # For non-strict mode, allow more flexibility
    if not strict_mode:
        # Minimum length for any input
        if len(question) < 2:
            return False
        if len(question) > 500:
            return False
        return True

    # Strict mode validation for domain QA
    if len(question) < 10:
        return False

    if len(question) > 300:
        return False

    # Should not contain formatting artifacts
    forbidden_patterns = [
        r'answer:', r'a:', r'question-answer', r'generate', r'create',
        r'based on', r'text:', r'rules:', r'format as', r'complete question',
        r'natural and conversational', r'helpful and accurate'
    ]

    question_lower = question.lower()
    for pattern in forbidden_patterns:
        if re.search(pattern, question_lower):
            return False

    # Should have reasonable word count
    words = question.split()
    if len(words) < 2:
        return False

    if len(words) > 50:
        return False

    return True


def is_valid_answer(answer: str, strict_mode: bool = True) -> bool:
    """
    Validation for answers.

    Args:
        answer: The answer text
        strict_mode: If True, apply strict validation. If False, allow shorter responses.
    """
    answer = answer.strip()

    # Basic checks
    if not answer:
        return False

    # For conversational responses, allow shorter answers
    if not strict_mode:
        if len(answer) < 5:
            return False
        if len(answer) > 1000:
            return False
        return True

    # Strict mode validation
    if len(answer) < 15:
        return False

    if len(answer) > 800:
        return False

    # Should not end with incomplete phrases
    incomplete_endings = [
        ' of', ' and', ' the', ' that', ' which', ' is', ' are', ' was', ' were',
        ' in', ' on', ' at', ' to', ' for', ' with', ' by', ' from', ' a', ' an',
    ]

    for ending in incomplete_endings:
        if answer.endswith(ending):
            return False

    # Should not contain formatting artifacts
    forbidden_patterns = [
        r'question:', r'q:', r'question-answer', r'generate', r'create',
        r'based on', r'text:', r'rules:', r'format as'
    ]

    answer_lower = answer.lower()
    for pattern in forbidden_patterns:
        if re.search(pattern, answer_lower):
            return False

    # Should have reasonable word count
    words = answer.split()
    if len(words) < 3:
        return False

    if len(words) > 150:
        return False

    return True


class DuplicateChecker:
    """
    Efficient duplicate checker using hash-based exact matching
    and Jaccard similarity for fuzzy matching.

    Uses O(1) hash lookup for exact matches, falling back to
    similarity check only when needed.
    """

    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self.exact_hashes: set = set()
        self.seen_pairs: List[Dict[str, str]] = []

    def _get_hash(self, question: str, answer: str) -> str:
        """Generate hash for exact match detection."""
        return f"{question.lower().strip()}|||{answer.lower().strip()}"

    def is_duplicate(self, qa_pair: Dict[str, str]) -> bool:
        """Check if QA pair is duplicate using hash + similarity."""
        question = qa_pair['question'].lower().strip()
        answer = qa_pair['answer'].lower().strip()

        # Fast O(1) exact match check
        pair_hash = self._get_hash(question, answer)
        if pair_hash in self.exact_hashes:
            return True

        # Similarity check against recent pairs (limit to avoid O(n²))
        # Only check last 100 pairs for similarity to bound complexity
        check_limit = min(100, len(self.seen_pairs))
        for seen_pair in self.seen_pairs[-check_limit:]:
            seen_question = seen_pair['question'].lower().strip()
            seen_answer = seen_pair['answer'].lower().strip()

            if (similarity_score(question, seen_question) > self.similarity_threshold and
                similarity_score(answer, seen_answer) > self.similarity_threshold):
                return True

        return False

    def add(self, qa_pair: Dict[str, str]) -> None:
        """Add a pair to the seen set."""
        question = qa_pair['question']
        answer = qa_pair['answer']
        pair_hash = self._get_hash(question, answer)
        self.exact_hashes.add(pair_hash)
        self.seen_pairs.append(qa_pair)


def is_duplicate(qa_pair: Dict[str, str], seen_pairs: List[Dict[str, str]],
                 similarity_threshold: float = 0.8) -> bool:
    """
    Check if QA pair is duplicate or very similar.

    Note: For better performance with large datasets, use DuplicateChecker class.
    """
    question = qa_pair['question'].lower().strip()
    answer = qa_pair['answer'].lower().strip()

    # Create hash for O(1) exact match check
    pair_hash = f"{question}|||{answer}"

    for seen_pair in seen_pairs:
        seen_question = seen_pair['question'].lower().strip()
        seen_answer = seen_pair['answer'].lower().strip()
        seen_hash = f"{seen_question}|||{seen_answer}"

        # Exact match (fast)
        if pair_hash == seen_hash:
            return True

        # High similarity (slower, but still needed for fuzzy dedup)
        if (similarity_score(question, seen_question) > similarity_threshold and
            similarity_score(answer, seen_answer) > similarity_threshold):
            return True

    return False


def similarity_score(str1: str, str2: str) -> float:
    """Calculate simple Jaccard similarity score between two strings."""
    if not str1 or not str2:
        return 0.0

    words1 = set(str1.split())
    words2 = set(str2.split())

    if not words1 or not words2:
        return 0.0

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    return len(intersection) / len(union)


def clean_text(text: str) -> str:
    """Clean text by removing artifacts."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove formatting artifacts
    text = re.sub(r'^(Q:|A:|Question:|Answer:)\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\d+\.\s*', '', text)

    # Clean up punctuation spacing
    text = re.sub(r'\s+([,.!?])', r'\1', text)

    return text.strip()


def clean_qa_dataset(
    data: List[Dict[str, Any]],
    strict_mode: bool = True,
    skip_duplicates: bool = True,
    similarity_threshold: float = 0.8
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Clean QA dataset with configurable strictness.

    Args:
        data: List of QA pairs
        strict_mode: If True, apply strict validation for domain QA.
                    If False, allow conversational and command-style inputs.
        skip_duplicates: If True, remove duplicate entries.
        similarity_threshold: Threshold for duplicate detection (0.0-1.0)

    Returns:
        Tuple of (cleaned_data, stats_dict)
    """
    cleaned_data = []
    stats = defaultdict(int)

    # Use efficient DuplicateChecker for O(1) exact match + bounded similarity
    dup_checker = DuplicateChecker(similarity_threshold=similarity_threshold)

    for item in data:
        stats['total'] += 1

        # Check for required fields
        question_key = 'question' if 'question' in item else 'input'
        answer_key = 'answer' if 'answer' in item else 'output'

        if question_key not in item or answer_key not in item:
            stats['missing_fields'] += 1
            continue

        # Clean the text
        question = clean_text(str(item[question_key]))
        answer = clean_text(str(item[answer_key]))

        # Check if it's conversational (use relaxed validation)
        is_conversational = is_conversational_input(question)
        use_strict = strict_mode and not is_conversational

        # Validate question
        if not is_valid_question(question, strict_mode=use_strict):
            stats['invalid_question'] += 1
            continue

        # Validate answer
        if not is_valid_answer(answer, strict_mode=use_strict):
            stats['invalid_answer'] += 1
            continue

        # Check for duplicates using efficient checker
        qa_pair = {'question': question, 'answer': answer}
        if skip_duplicates and dup_checker.is_duplicate(qa_pair):
            stats['duplicate'] += 1
            continue

        # Add to cleaned data
        cleaned_item = item.copy()
        cleaned_item['question'] = question
        cleaned_item['answer'] = answer

        if is_conversational:
            cleaned_item['type'] = 'conversational'
            stats['conversational'] += 1
        else:
            stats['qa'] += 1

        cleaned_data.append(cleaned_item)
        dup_checker.add(qa_pair)
        stats['valid'] += 1

    return cleaned_data, stats


def print_stats(stats: Dict[str, int]) -> None:
    """Print cleaning statistics."""
    print("\n" + "=" * 50)
    print("DATA CLEANING STATISTICS")
    print("=" * 50)

    total = stats['total']
    valid = stats['valid']

    print(f"Total entries processed: {total}")
    print(f"Valid entries kept: {valid}")
    print(f"Entries removed: {total - valid}")

    if total > 0:
        print(f"Retention rate: {(valid/total)*100:.1f}%")

    print("\nBy type:")
    print(f"  QA pairs: {stats.get('qa', 0)}")
    print(f"  Conversational: {stats.get('conversational', 0)}")

    print("\nRemoval breakdown:")
    print(f"  Missing fields: {stats['missing_fields']}")
    print(f"  Invalid questions: {stats['invalid_question']}")
    print(f"  Invalid answers: {stats['invalid_answer']}")
    print(f"  Duplicates: {stats['duplicate']}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Clean QA dataset for chatbot training")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    parser.add_argument(
        "--relaxed",
        action="store_true",
        help="Use relaxed validation (allows shorter inputs, conversational phrases)"
    )
    parser.add_argument(
        "--keep_duplicates",
        action="store_true",
        help="Keep duplicate entries"
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed statistics")

    args = parser.parse_args()

    print(f"Loading data from: {args.input}")
    data = load_jsonl(args.input)
    print(f"Loaded {len(data)} entries")

    mode = "relaxed" if args.relaxed else "strict"
    print(f"Cleaning dataset in {mode} mode...")

    cleaned_data, stats = clean_qa_dataset(
        data,
        strict_mode=not args.relaxed,
        skip_duplicates=not args.keep_duplicates
    )

    print(f"Saving cleaned data to: {args.output}")
    save_jsonl(cleaned_data, args.output)

    print_stats(stats)

    if args.verbose and cleaned_data:
        print("\nSample of cleaned entries:")
        for i, item in enumerate(cleaned_data[:3]):
            print(f"\nEntry {i+1}:")
            print(f"Q: {item.get('question', item.get('input', 'N/A'))}")
            print(f"A: {item.get('answer', item.get('output', 'N/A'))}")


if __name__ == "__main__":
    main()

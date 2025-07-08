"""
Enhanced Data Cleaner for QA Dataset

This script cleans malformed QA pairs from JSONL files, removing:
- Incomplete answers
- Malformed questions
- Duplicate entries
- Entries with prompt leakage
- Low-quality content unsuitable for chatbot training
"""

import json
import argparse
from typing import List, Dict, Any
import re
from collections import defaultdict


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
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def is_valid_question(question: str) -> bool:
    """Enhanced validation for questions suitable for chatbot training."""
    question = question.strip()
    
    # Basic checks
    if not question:
        return False
    
    # Length validation - questions should be substantial
    if len(question) < 15:
        return False
    
    if len(question) > 300:
        return False
    
    # Must end with question mark
    if not question.endswith('?'):
        return False
    
    # Should not contain answer markers or formatting artifacts
    forbidden_patterns = [
        r'answer:', r'a:', r'question-answer', r'generate', r'create',
        r'based on', r'text:', r'rules:', r'format as', r'complete question',
        r'natural and conversational', r'helpful and accurate'
    ]
    
    question_lower = question.lower()
    for pattern in forbidden_patterns:
        if re.search(pattern, question_lower):
            return False
    
    # Should not start with formatting artifacts
    if re.match(r'^[0-9]+\.\s*[^?]*[^?]$', question):
        return False
    
    # Should not be repetitive or nonsensical
    words = question.split()
    if len(words) < 3:
        return False
    
    # Check for reasonable word count
    if len(words) > 50:
        return False
    
    # Should not be too similar to common phrases
    common_phrases = [
        'what is', 'how does', 'why is', 'when is', 'where is',
        'can you', 'could you', 'would you', 'should you'
    ]
    
    question_lower = question.lower()
    if any(phrase in question_lower for phrase in common_phrases):
        # Only allow if there's substantial content beyond the common phrase
        if len(question) < 25:
            return False
    
    return True


def is_valid_answer(answer: str) -> bool:
    """Enhanced validation for answers suitable for chatbot training."""
    answer = answer.strip()
    
    # Basic checks
    if not answer:
        return False
    
    # Length validation - answers should be substantial
    if len(answer) < 20:
        return False
    
    if len(answer) > 800:
        return False
    
    # Should not end with incomplete phrases
    incomplete_endings = [
        ' of', ' and', ' the', ' that', ' which', ' is', ' are', ' was', ' were',
        ' in', ' on', ' at', ' to', ' for', ' with', ' by', ' from', ' a', ' an',
        ' this', ' these', ' those', ' it', ' they', ' them', ' their'
    ]
    
    for ending in incomplete_endings:
        if answer.endswith(ending):
            return False
    
    # Should not contain question markers or formatting artifacts
    forbidden_patterns = [
        r'question:', r'q:', r'question-answer', r'generate', r'create',
        r'based on', r'text:', r'rules:', r'format as', r'complete answer',
        r'natural and conversational', r'helpful and accurate'
    ]
    
    answer_lower = answer.lower()
    for pattern in forbidden_patterns:
        if re.search(pattern, answer_lower):
            return False
    
    # Should end with proper punctuation
    if not answer.endswith(('.', '!', '?')):
        return False
    
    # Should have reasonable word count
    words = answer.split()
    if len(words) < 5:
        return False
    
    if len(words) > 150:
        return False
    
    # Should not be repetitive or nonsensical
    if len(set(words)) < len(words) * 0.3:  # Too repetitive
        return False
    
    return True


def is_duplicate(qa_pair: Dict[str, str], seen_pairs: List[Dict[str, str]], 
                similarity_threshold: float = 0.7) -> bool:
    """Check if QA pair is duplicate or very similar."""
    question = qa_pair['question'].lower().strip()
    answer = qa_pair['answer'].lower().strip()
    
    for seen_pair in seen_pairs:
        seen_question = seen_pair['question'].lower().strip()
        seen_answer = seen_pair['answer'].lower().strip()
        
        # Exact match
        if question == seen_question and answer == seen_answer:
            return True
        
        # High similarity (more strict threshold)
        if (similarity_score(question, seen_question) > similarity_threshold and 
            similarity_score(answer, seen_answer) > similarity_threshold):
            return True
    
    return False


def similarity_score(str1: str, str2: str) -> float:
    """Calculate simple similarity score between two strings."""
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
    """Enhanced text cleaning for chatbot training."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove formatting artifacts more thoroughly
    text = re.sub(r'^(Q:|A:|Question:|Answer:)\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\d+\.\s*', '', text)
    
    # Remove common artifacts
    text = re.sub(r'^(Rules:|Format:|Create|Generate)\s*', '', text, flags=re.IGNORECASE)
    
    # Clean up punctuation
    text = re.sub(r'\s+([,.!?])', r'\1', text)
    
    # Remove trailing punctuation artifacts
    text = re.sub(r'\s*[.]{2,}\s*$', '', text)
    
    return text.strip()


def is_complete_qa_pair(qa_pair: Dict[str, str]) -> bool:
    """Additional validation for completeness of QA pairs."""
    question = qa_pair['question']
    answer = qa_pair['answer']
    
    # Check for balanced sentence structure
    if question.count('?') != 1:
        return False
    
    # Check for reasonable answer length relative to question
    if len(answer) < len(question) * 0.5:
        return False
    
    # Check for proper sentence endings in answer
    if not answer.endswith(('.', '!', '?')):
        return False
    
    # Check for reasonable word count
    question_words = len(question.split())
    answer_words = len(answer.split())
    
    if question_words < 3 or answer_words < 5:
        return False
    
    # Check that answer doesn't just repeat the question
    question_lower = question.lower().replace('?', '').strip()
    answer_lower = answer.lower()
    
    if question_lower in answer_lower and len(answer_lower) < len(question_lower) * 2:
        return False
    
    return True


def clean_qa_dataset(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Enhanced cleaning of QA dataset for chatbot training."""
    cleaned_data = []
    seen_pairs = []
    stats = defaultdict(int)
    
    for item in data:
        stats['total'] += 1
        
        # Check if required fields exist
        if 'question' not in item or 'answer' not in item:
            stats['missing_fields'] += 1
            continue
        
        # Clean the text
        question = clean_text(item['question'])
        answer = clean_text(item['answer'])
        
        # Validate question
        if not is_valid_question(question):
            stats['invalid_question'] += 1
            continue
        
        # Validate answer
        if not is_valid_answer(answer):
            stats['invalid_answer'] += 1
            continue
        
        # Check for duplicates
        qa_pair = {'question': question, 'answer': answer}
        if is_duplicate(qa_pair, seen_pairs):
            stats['duplicate'] += 1
            continue
        
        # Additional completeness check
        if not is_complete_qa_pair(qa_pair):
            stats['incomplete'] += 1
            continue
        
        # Add to cleaned data
        cleaned_item = item.copy()
        cleaned_item['question'] = question
        cleaned_item['answer'] = answer
        
        cleaned_data.append(cleaned_item)
        seen_pairs.append(qa_pair)
        stats['valid'] += 1
    
    return cleaned_data, stats


def print_stats(stats: Dict[str, int]) -> None:
    """Print cleaning statistics."""
    print("\n" + "="*50)
    print("DATA CLEANING STATISTICS")
    print("="*50)
    
    total = stats['total']
    valid = stats['valid']
    
    print(f"Total entries processed: {total}")
    print(f"Valid entries kept: {valid}")
    print(f"Entries removed: {total - valid}")
    print(f"Retention rate: {(valid/total)*100:.1f}%")
    
    print("\nRemoval breakdown:")
    print(f"  Missing fields: {stats['missing_fields']}")
    print(f"  Invalid questions: {stats['invalid_question']}")
    print(f"  Invalid answers: {stats['invalid_answer']}")
    print(f"  Duplicates: {stats['duplicate']}")
    print(f"  Incomplete pairs: {stats['incomplete']}")


def main():
    parser = argparse.ArgumentParser(description="Clean QA dataset for chatbot training")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--verbose", action="store_true", help="Print detailed statistics")
    
    args = parser.parse_args()
    
    print(f"Loading data from: {args.input}")
    data = load_jsonl(args.input)
    print(f"Loaded {len(data)} entries")
    
    print("Cleaning dataset...")
    cleaned_data, stats = clean_qa_dataset(data)
    
    print(f"Saving cleaned data to: {args.output}")
    save_jsonl(cleaned_data, args.output)
    
    print_stats(stats)
    
    if args.verbose:
        print("\nSample of cleaned entries:")
        for i, item in enumerate(cleaned_data[:3]):
            print(f"\nEntry {i+1}:")
            print(f"Q: {item['question']}")
            print(f"A: {item['answer']}")


if __name__ == "__main__":
    main()
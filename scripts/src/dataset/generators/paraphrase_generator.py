"""
Paraphrase Generator for QA Dataset Augmentation

Generates multiple phrasings of the same question to improve model robustness
to differently phrased inputs.
"""

import random
import re
from typing import List, Dict, Optional


# Question prefix variations organized by question type
QUESTION_PREFIXES = {
    "what": [
        "What is", "What are", "What's",
        "Can you tell me what", "Could you explain what",
        "I'd like to know what", "Please tell me what",
        "Do you know what", "Help me understand what"
    ],
    "how": [
        "How does", "How do", "How can", "How would",
        "Can you explain how", "Could you describe how",
        "What's the way to", "In what way does",
        "Please explain how", "Tell me how"
    ],
    "why": [
        "Why is", "Why does", "Why do", "Why are",
        "What's the reason", "Can you explain why",
        "What causes", "What makes", "For what reason"
    ],
    "when": [
        "When is", "When does", "When do", "When will",
        "At what time", "Could you tell me when",
        "At what point", "During what time"
    ],
    "where": [
        "Where is", "Where are", "Where can", "Where do",
        "In what location", "Could you tell me where",
        "At what place", "In which place"
    ],
    "who": [
        "Who is", "Who are", "Who does", "Who was",
        "Can you tell me who", "Could you identify who",
        "Which person", "What person"
    ],
    "which": [
        "Which is", "Which are", "Which one",
        "Can you tell me which", "What specific"
    ],
    "can": [
        "Can you", "Could you", "Would you be able to",
        "Is it possible to", "Are you able to"
    ],
    "is": [
        "Is it", "Is there", "Is this",
        "Would it be", "Does it"
    ],
    "are": [
        "Are there", "Are they", "Are these",
        "Would there be", "Do they"
    ],
}

# Imperative/command variations (non-question format)
IMPERATIVE_TEMPLATES = [
    "Tell me about {topic}",
    "Explain {topic}",
    "Describe {topic}",
    "Give me information about {topic}",
    "I need to know about {topic}",
    "Help me understand {topic}",
    "Can you elaborate on {topic}",
    "I want to learn about {topic}",
    "Please explain {topic}",
    "Share information about {topic}",
    "Provide details about {topic}",
    "I'm curious about {topic}",
]


def extract_topic_from_question(question: str) -> str:
    """
    Extract the main topic/subject from a question.

    Examples:
        "What is machine learning?" -> "machine learning"
        "How does photosynthesis work?" -> "photosynthesis work"
    """
    question = question.strip().rstrip('?').lower()

    # Common question starters to remove
    starters = [
        'what is', 'what are', 'what\'s', 'what does', 'what do',
        'how does', 'how do', 'how can', 'how is', 'how are',
        'why is', 'why does', 'why do', 'why are',
        'when is', 'when does', 'when do', 'when was',
        'where is', 'where are', 'where can', 'where do',
        'who is', 'who are', 'who does', 'who was',
        'which is', 'which are',
        'can you tell me', 'could you explain', 'please tell me',
        'can you', 'could you', 'would you',
        'is there', 'are there', 'is it', 'are they',
        'tell me about', 'explain', 'describe',
    ]

    for starter in sorted(starters, key=len, reverse=True):
        if question.startswith(starter):
            topic = question[len(starter):].strip()
            # Remove leading articles
            for article in ['a ', 'an ', 'the ']:
                if topic.startswith(article):
                    topic = topic[len(article):]
            return topic

    return question


def get_question_type(question: str) -> Optional[str]:
    """Determine the type of question based on its starting word."""
    question_lower = question.lower().strip()

    for q_type in QUESTION_PREFIXES.keys():
        if question_lower.startswith(q_type):
            return q_type

    return None


def generate_paraphrases(question: str, num_variations: int = 3,
                         include_imperatives: bool = True) -> List[str]:
    """
    Generate paraphrased versions of a question.

    Args:
        question: Original question to paraphrase
        num_variations: Number of variations to generate
        include_imperatives: Whether to include command-style variations

    Returns:
        List of question variations (always includes original)
    """
    variations = [question]  # Always include original
    topic = extract_topic_from_question(question)
    question_type = get_question_type(question)

    # Generate question-style variations
    if question_type and question_type in QUESTION_PREFIXES:
        prefixes = QUESTION_PREFIXES[question_type]
        for prefix in prefixes:
            variation = f"{prefix} {topic}?"
            # Avoid duplicates (case-insensitive)
            if variation.lower() != question.lower() and variation not in variations:
                variations.append(variation)

    # Add imperative/command variations
    if include_imperatives:
        for template in IMPERATIVE_TEMPLATES:
            variation = template.format(topic=topic)
            if variation not in variations:
                variations.append(variation)

    # Shuffle and select requested number
    if len(variations) > num_variations + 1:
        # Keep original, shuffle rest
        rest = variations[1:]
        random.shuffle(rest)
        variations = [variations[0]] + rest[:num_variations]

    return variations


def augment_qa_pair(qa_pair: Dict[str, str],
                    num_variations: int = 3) -> List[Dict[str, str]]:
    """
    Augment a single QA pair with paraphrased questions.

    Args:
        qa_pair: Dictionary with 'question' and 'answer' keys
        num_variations: Number of variations to generate

    Returns:
        List of QA pairs with different question phrasings
    """
    original_question = qa_pair.get('question', '')
    answer = qa_pair.get('answer', '')

    if not original_question or not answer:
        return [qa_pair]

    # Generate question variations
    question_variations = generate_paraphrases(original_question, num_variations)

    # Create new pairs for each variation
    augmented = []
    for variation in question_variations:
        new_pair = qa_pair.copy()
        new_pair['question'] = variation
        new_pair['original_question'] = original_question
        augmented.append(new_pair)

    return augmented


def augment_qa_dataset(qa_pairs: List[Dict[str, str]],
                       variations_per_question: int = 3) -> List[Dict[str, str]]:
    """
    Augment entire QA dataset with paraphrased questions.

    Args:
        qa_pairs: List of QA pair dictionaries
        variations_per_question: Number of paraphrase variations per question

    Returns:
        Augmented list of QA pairs
    """
    augmented = []

    for pair in qa_pairs:
        augmented_pairs = augment_qa_pair(pair, variations_per_question)
        augmented.extend(augmented_pairs)

    return augmented


def augment_jsonl_file(input_path: str, output_path: str,
                       variations_per_question: int = 3) -> None:
    """
    Augment a JSONL file with paraphrased questions.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to save augmented JSONL file
        variations_per_question: Number of variations per question
    """
    import json
    from pathlib import Path

    # Load data
    qa_pairs = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    qa_pairs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    print(f"Loaded {len(qa_pairs)} QA pairs from {input_path}")

    # Augment
    augmented = augment_qa_dataset(qa_pairs, variations_per_question)
    print(f"Generated {len(augmented)} augmented pairs ({variations_per_question + 1}x expansion)")

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in augmented:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    print(f"Saved augmented dataset to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Augment QA dataset with paraphrases")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--variations", type=int, default=3, help="Variations per question")

    args = parser.parse_args()

    augment_jsonl_file(args.input, args.output, args.variations)

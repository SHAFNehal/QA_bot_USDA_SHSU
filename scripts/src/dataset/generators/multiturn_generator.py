"""
Multi-Turn Conversation Data Generator

Generates follow-up question chains for training conversational ability.
This helps the model understand pronouns and maintain context across turns.
"""

import json
import random
import argparse
from typing import List, Dict, Optional
from pathlib import Path


# Follow-up question templates organized by type
FOLLOWUP_TEMPLATES = {
    "clarification": [
        "Can you explain that more?",
        "What do you mean by that?",
        "Could you elaborate?",
        "Can you clarify that?",
        "I'm not sure I understand. Can you explain?",
    ],
    "detail": [
        "What are the main steps?",
        "How does that work exactly?",
        "What are the key components?",
        "Can you break that down?",
        "What are the details?",
    ],
    "example": [
        "Can you give me an example?",
        "What's a practical example of this?",
        "Could you provide an example?",
        "Can you show me how that works?",
    ],
    "pronoun": [
        "How efficient is it?",
        "When was it discovered?",
        "Who invented it?",
        "Why is it important?",
        "What are its limitations?",
        "What are its benefits?",
        "How is it used?",
        "Where is it applied?",
    ],
    "continuation": [
        "What else should I know?",
        "Is there anything else?",
        "What's next?",
        "And then what?",
        "Can you tell me more?",
    ],
}

# Coreference examples - conversations that explicitly teach pronoun resolution
COREFERENCE_EXAMPLES = [
    {
        "turns": [
            {"user": "What is machine learning?", "assistant": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."},
            {"user": "How does it work?", "assistant": "It works by using algorithms to identify patterns in data and make predictions or decisions based on those patterns. The system improves its accuracy over time as it processes more data."},
            {"user": "What are its applications?", "assistant": "Its applications include image recognition, natural language processing, recommendation systems, autonomous vehicles, fraud detection, and medical diagnosis."}
        ]
    },
    {
        "turns": [
            {"user": "Tell me about the sun.", "assistant": "The sun is a star at the center of our solar system. It's a nearly perfect sphere of hot plasma that provides light and heat to Earth."},
            {"user": "How hot is it?", "assistant": "The sun's surface temperature is about 5,500 degrees Celsius, while its core reaches approximately 15 million degrees Celsius."},
            {"user": "Why is it important?", "assistant": "It is important because it provides the light and heat necessary for life on Earth. It drives weather patterns, enables photosynthesis, and is the primary source of energy for our planet."}
        ]
    },
    {
        "turns": [
            {"user": "What is photosynthesis?", "assistant": "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen. It's how plants make their own food."},
            {"user": "Where does it occur?", "assistant": "It occurs primarily in the leaves of plants, specifically in cell structures called chloroplasts which contain the green pigment chlorophyll."},
            {"user": "How efficient is it?", "assistant": "It is typically 3-6% efficient at converting solar energy into chemical energy. While this seems low, it's remarkably effective for sustaining life on Earth."}
        ]
    },
    {
        "turns": [
            {"user": "What is DNA?", "assistant": "DNA, or deoxyribonucleic acid, is a molecule that carries the genetic instructions for all living organisms. It contains the blueprint for building and maintaining an organism."},
            {"user": "What does it look like?", "assistant": "It has a double helix structure, resembling a twisted ladder. The 'rungs' are made of base pairs (adenine-thymine and guanine-cytosine) while the 'sides' are sugar-phosphate backbones."},
            {"user": "Where is it found?", "assistant": "It is found in the nucleus of cells in most organisms. In humans, DNA is organized into 23 pairs of chromosomes within each cell's nucleus."}
        ]
    },
    {
        "turns": [
            {"user": "What is Python?", "assistant": "Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used for web development, data science, AI, and automation."},
            {"user": "Why is it popular?", "assistant": "It is popular because of its clean syntax, extensive libraries, strong community support, and versatility. It's easy to learn for beginners yet powerful enough for complex applications."},
            {"user": "What can you build with it?", "assistant": "You can build web applications, data analysis tools, machine learning models, automation scripts, games, and much more. Popular frameworks include Django, Flask, and TensorFlow."}
        ]
    },
]


def get_random_followup(followup_type: str = None) -> str:
    """Get a random follow-up question template."""
    if followup_type and followup_type in FOLLOWUP_TEMPLATES:
        return random.choice(FOLLOWUP_TEMPLATES[followup_type])

    # Random type
    all_followups = []
    for templates in FOLLOWUP_TEMPLATES.values():
        all_followups.extend(templates)
    return random.choice(all_followups)


def format_conversation_for_training(
    conversation: List[Dict[str, str]],
    system_prompt: str = "You are a helpful, friendly assistant."
) -> List[Dict[str, str]]:
    """
    Format a multi-turn conversation into training examples.

    Each example includes the full conversation history up to that point.
    This teaches the model to understand context from previous turns.
    """
    training_examples = []

    for i in range(len(conversation)):
        # Build history up to current turn
        history = ""
        for j in range(i):
            turn = conversation[j]
            history += f"<|user|>\n{turn['user']}</s>\n"
            history += f"<|assistant|>\n{turn['assistant']}</s>\n"

        current_turn = conversation[i]

        # Format as training example
        example = {
            'input': history + f"<|user|>\n{current_turn['user']}</s>\n<|assistant|>\n",
            'output': current_turn['assistant'],
            'turn_number': i + 1,
            'total_turns': len(conversation),
            'type': 'multiturn'
        }

        training_examples.append(example)

    return training_examples


def generate_coreference_training_data() -> List[Dict[str, str]]:
    """
    Generate training data from predefined coreference examples.

    These examples explicitly teach the model to resolve pronouns
    like "it", "its", "they", etc.
    """
    training_data = []

    for example in COREFERENCE_EXAMPLES:
        turns = example['turns']
        formatted = format_conversation_for_training(turns)
        training_data.extend(formatted)

    return training_data


def create_multiturn_from_qa(
    qa_pair: Dict[str, str],
    num_followups: int = 2
) -> List[Dict[str, str]]:
    """
    Create a multi-turn conversation from a single QA pair.

    Uses the original QA as the first turn, then adds follow-up questions
    with placeholder responses (to be filled by actual model generation).
    """
    conversation = [
        {
            'user': qa_pair.get('question', ''),
            'assistant': qa_pair.get('answer', '')
        }
    ]

    # Add follow-up turns with generic but helpful responses
    followup_responses = [
        "I'd be happy to provide more details. What specific aspect would you like me to explain further?",
        "That's a great follow-up question. Let me elaborate on that point.",
        "I can explain more about that. Is there a particular part you'd like me to focus on?",
    ]

    for i in range(num_followups):
        followup_type = random.choice(list(FOLLOWUP_TEMPLATES.keys()))
        followup = get_random_followup(followup_type)

        conversation.append({
            'user': followup,
            'assistant': random.choice(followup_responses)
        })

    return conversation


def augment_dataset_with_coreference(
    qa_pairs: List[Dict[str, str]],
    include_predefined: bool = True
) -> List[Dict[str, str]]:
    """
    Augment QA dataset with multi-turn coreference examples.

    Args:
        qa_pairs: Original QA pairs
        include_predefined: Include predefined coreference examples

    Returns:
        Combined list of single-turn and multi-turn training examples
    """
    all_examples = []

    # Add original single-turn examples in multi-turn format
    for qa in qa_pairs:
        if 'question' in qa and 'answer' in qa:
            example = {
                'input': f"<|user|>\n{qa['question']}</s>\n<|assistant|>\n",
                'output': qa['answer'],
                'turn_number': 1,
                'total_turns': 1,
                'type': 'single_turn'
            }
            all_examples.append(example)

    # Add predefined coreference examples
    if include_predefined:
        coreference_data = generate_coreference_training_data()
        all_examples.extend(coreference_data)
        print(f"Added {len(coreference_data)} coreference training examples")

    return all_examples


def save_multiturn_dataset(data: List[Dict], output_path: str) -> None:
    """Save multi-turn dataset to JSONL file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved {len(data)} examples to {output_path}")


def load_qa_pairs(input_path: str) -> List[Dict]:
    """Load QA pairs from JSONL file."""
    pairs = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    pairs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Generate multi-turn conversation data")

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input QA dataset (JSONL)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data_output/multiturn_dataset.jsonl",
        help="Output path for multi-turn dataset"
    )

    parser.add_argument(
        "--include_coreference",
        action="store_true",
        default=True,
        help="Include predefined coreference examples"
    )

    parser.add_argument(
        "--coreference_only",
        action="store_true",
        help="Only generate coreference examples (no input QA needed)"
    )

    args = parser.parse_args()

    all_examples = []

    # Load input QA pairs if provided
    if args.input and not args.coreference_only:
        if Path(args.input).exists():
            qa_pairs = load_qa_pairs(args.input)
            print(f"Loaded {len(qa_pairs)} QA pairs from {args.input}")
            augmented = augment_dataset_with_coreference(qa_pairs, args.include_coreference)
            all_examples.extend(augmented)
        else:
            print(f"Warning: Input file {args.input} not found")

    # Generate coreference examples
    if args.coreference_only or (not args.input and args.include_coreference):
        coreference_data = generate_coreference_training_data()
        all_examples.extend(coreference_data)
        print(f"Generated {len(coreference_data)} coreference examples")

    if all_examples:
        save_multiturn_dataset(all_examples, args.output)
        print(f"\nTotal examples: {len(all_examples)}")
    else:
        print("No examples generated. Provide --input or use --coreference_only")


if __name__ == "__main__":
    main()

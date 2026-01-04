"""
Pytest fixtures and configuration for the test suite.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_qa_pairs():
    """Sample QA pairs for testing."""
    return [
        {
            "question": "What is machine learning?",
            "answer": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."
        },
        {
            "question": "How does photosynthesis work?",
            "answer": "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen using chlorophyll in their leaves."
        },
        {
            "question": "What is the capital of France?",
            "answer": "The capital of France is Paris, which is also the country's largest city and a major European cultural and economic center."
        }
    ]


@pytest.fixture
def sample_conversational_inputs():
    """Sample conversational inputs for testing."""
    return [
        "Hi",
        "Hello there",
        "Good morning",
        "Thanks",
        "Thank you so much",
        "Bye",
        "Goodbye",
        "Okay",
        "Got it",
        "Who are you?",
        "What can you do?",
    ]


@pytest.fixture
def sample_multiturn_conversation():
    """Sample multi-turn conversation for testing."""
    return [
        {"user": "What is Python?", "assistant": "Python is a high-level programming language known for its simplicity and readability."},
        {"user": "Why is it popular?", "assistant": "It's popular due to its clean syntax, extensive libraries, and versatility across domains like web development, data science, and AI."},
        {"user": "What can you build with it?", "assistant": "You can build web applications, data analysis tools, machine learning models, automation scripts, and much more."}
    ]


@pytest.fixture
def sample_dirty_qa_pairs():
    """Sample QA pairs with various quality issues for testing cleaning."""
    return [
        # Valid pair
        {"question": "What is the speed of light?", "answer": "The speed of light in a vacuum is approximately 299,792 kilometers per second."},
        # Too short question
        {"question": "Hi?", "answer": "Hello there!"},
        # Missing question mark (for strict mode)
        {"question": "Tell me about gravity", "answer": "Gravity is a fundamental force that attracts objects with mass toward each other."},
        # Too short answer
        {"question": "What is water?", "answer": "H2O"},
        # Duplicate
        {"question": "What is the speed of light?", "answer": "The speed of light in a vacuum is approximately 299,792 kilometers per second."},
        # Conversational input (should pass with relaxed mode)
        {"question": "Hello", "answer": "Hi there! How can I help you today?"},
    ]

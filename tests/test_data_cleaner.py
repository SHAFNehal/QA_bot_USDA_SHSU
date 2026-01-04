"""
Unit tests for data_cleaner.py
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.data_cleaner import (
    is_conversational_input,
    is_valid_conversational,
    is_valid_question,
    is_valid_answer,
    is_duplicate,
    similarity_score,
    clean_text,
    clean_qa_dataset,
    DuplicateChecker,
)


class TestIsConversationalInput:
    """Tests for is_conversational_input function."""

    def test_greetings_are_conversational(self, sample_conversational_inputs):
        """Test that greetings are detected as conversational."""
        greetings = ["Hi", "Hello", "Hey", "Good morning", "Hi there"]
        for greeting in greetings:
            assert is_conversational_input(greeting), f"'{greeting}' should be conversational"

    def test_thanks_are_conversational(self):
        """Test that thanks are detected as conversational."""
        thanks = ["Thanks", "Thank you", "Thanks a lot", "I appreciate it"]
        for thank in thanks:
            assert is_conversational_input(thank), f"'{thank}' should be conversational"

    def test_farewells_are_conversational(self):
        """Test that farewells are detected as conversational."""
        farewells = ["Bye", "Goodbye", "See you", "Take care"]
        for farewell in farewells:
            assert is_conversational_input(farewell), f"'{farewell}' should be conversational"

    def test_regular_questions_are_not_conversational(self):
        """Test that regular questions are not conversational."""
        questions = [
            "What is machine learning?",
            "How does photosynthesis work?",
            "Why is the sky blue?",
        ]
        for q in questions:
            assert not is_conversational_input(q), f"'{q}' should not be conversational"


class TestIsValidConversational:
    """Tests for is_valid_conversational function."""

    def test_valid_short_inputs(self):
        """Test that valid short inputs pass."""
        assert is_valid_conversational("Hi")
        assert is_valid_conversational("Hello")
        assert is_valid_conversational("Thanks")

    def test_empty_input_fails(self):
        """Test that empty input fails."""
        assert not is_valid_conversational("")
        assert not is_valid_conversational("   ")

    def test_too_long_input_fails(self):
        """Test that too long input fails."""
        long_input = "a" * 150
        assert not is_valid_conversational(long_input)


class TestIsValidQuestion:
    """Tests for is_valid_question function."""

    def test_valid_questions_pass_strict(self):
        """Test that valid questions pass in strict mode."""
        valid_questions = [
            "What is the capital of France?",
            "How does machine learning work?",
            "Why is the sky blue?",
        ]
        for q in valid_questions:
            assert is_valid_question(q, strict_mode=True), f"'{q}' should be valid"

    def test_short_questions_fail_strict(self):
        """Test that short questions fail in strict mode."""
        assert not is_valid_question("What?", strict_mode=True)
        assert not is_valid_question("How?", strict_mode=True)

    def test_too_long_questions_fail(self):
        """Test that too long questions fail."""
        long_q = "What is " + "a" * 400 + "?"
        assert not is_valid_question(long_q, strict_mode=True)

    def test_conversational_inputs_pass(self):
        """Test that conversational inputs pass validation."""
        assert is_valid_question("Hi", strict_mode=True)
        assert is_valid_question("Hello", strict_mode=True)

    def test_relaxed_mode_allows_more(self):
        """Test that relaxed mode allows more inputs."""
        # Short command-style input
        assert is_valid_question("Define AI", strict_mode=False)


class TestIsValidAnswer:
    """Tests for is_valid_answer function."""

    def test_valid_answers_pass_strict(self):
        """Test that valid answers pass in strict mode."""
        valid_answers = [
            "The capital of France is Paris, a major European city.",
            "Machine learning is a subset of AI that enables systems to learn from data.",
        ]
        for a in valid_answers:
            assert is_valid_answer(a, strict_mode=True), f"'{a}' should be valid"

    def test_short_answers_fail_strict(self):
        """Test that short answers fail in strict mode."""
        assert not is_valid_answer("Yes", strict_mode=True)
        assert not is_valid_answer("No", strict_mode=True)

    def test_incomplete_endings_fail(self):
        """Test that answers with incomplete endings fail."""
        incomplete = [
            "The answer is the",
            "It works by using the",
            "This is because of",
        ]
        for a in incomplete:
            assert not is_valid_answer(a, strict_mode=True), f"'{a}' should fail"

    def test_relaxed_mode_allows_shorter(self):
        """Test that relaxed mode allows shorter answers."""
        assert is_valid_answer("Hello there!", strict_mode=False)


class TestSimilarityScore:
    """Tests for similarity_score function."""

    def test_identical_strings(self):
        """Test that identical strings have score 1.0."""
        assert similarity_score("hello world", "hello world") == 1.0

    def test_completely_different_strings(self):
        """Test that completely different strings have score 0."""
        assert similarity_score("hello", "goodbye") == 0.0

    def test_partial_overlap(self):
        """Test that partial overlap gives intermediate score."""
        score = similarity_score("hello world", "hello there")
        assert 0 < score < 1

    def test_empty_strings(self):
        """Test handling of empty strings."""
        assert similarity_score("", "hello") == 0.0
        assert similarity_score("hello", "") == 0.0
        assert similarity_score("", "") == 0.0


class TestDuplicateChecker:
    """Tests for DuplicateChecker class."""

    def test_detects_exact_duplicates(self):
        """Test that exact duplicates are detected."""
        checker = DuplicateChecker()
        qa1 = {"question": "What is AI?", "answer": "AI is artificial intelligence."}

        checker.add(qa1)
        assert checker.is_duplicate(qa1)

    def test_allows_different_pairs(self):
        """Test that different pairs are allowed."""
        checker = DuplicateChecker()
        qa1 = {"question": "What is AI?", "answer": "AI is artificial intelligence."}
        qa2 = {"question": "What is ML?", "answer": "ML is machine learning."}

        checker.add(qa1)
        assert not checker.is_duplicate(qa2)

    def test_detects_similar_pairs(self):
        """Test that similar pairs are detected."""
        checker = DuplicateChecker(similarity_threshold=0.8)
        qa1 = {"question": "What is artificial intelligence?", "answer": "AI is a field of computer science."}
        qa2 = {"question": "What is artificial intelligence?", "answer": "AI is a field of computer science."}

        checker.add(qa1)
        assert checker.is_duplicate(qa2)


class TestIsDuplicate:
    """Tests for is_duplicate function."""

    def test_exact_duplicate(self):
        """Test detection of exact duplicates."""
        qa = {"question": "What is AI?", "answer": "AI is artificial intelligence."}
        seen = [qa]
        assert is_duplicate(qa, seen)

    def test_not_duplicate(self):
        """Test that different pairs are not duplicates."""
        qa1 = {"question": "What is AI?", "answer": "AI is artificial intelligence."}
        qa2 = {"question": "What is ML?", "answer": "ML is machine learning."}
        assert not is_duplicate(qa2, [qa1])

    def test_case_insensitive(self):
        """Test that duplicate check is case insensitive."""
        qa1 = {"question": "What is AI?", "answer": "AI is artificial intelligence."}
        qa2 = {"question": "WHAT IS AI?", "answer": "ai is artificial intelligence."}
        assert is_duplicate(qa2, [qa1])


class TestCleanText:
    """Tests for clean_text function."""

    def test_removes_extra_whitespace(self):
        """Test that extra whitespace is removed."""
        assert clean_text("hello   world") == "hello world"
        assert clean_text("  hello  ") == "hello"

    def test_removes_qa_prefixes(self):
        """Test that Q:/A: prefixes are removed."""
        assert clean_text("Q: What is AI?") == "What is AI?"
        assert clean_text("A: It is artificial intelligence.") == "It is artificial intelligence."

    def test_removes_numbering(self):
        """Test that numbering is removed."""
        assert clean_text("1. What is AI?") == "What is AI?"

    def test_fixes_punctuation_spacing(self):
        """Test that punctuation spacing is fixed."""
        assert clean_text("Hello , world !") == "Hello, world!"


class TestCleanQADataset:
    """Tests for clean_qa_dataset function."""

    def test_returns_tuple(self, sample_dirty_qa_pairs):
        """Test that function returns tuple of data and stats."""
        result = clean_qa_dataset(sample_dirty_qa_pairs)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_removes_invalid_pairs(self, sample_dirty_qa_pairs):
        """Test that invalid pairs are removed."""
        cleaned, stats = clean_qa_dataset(sample_dirty_qa_pairs, strict_mode=True)
        assert len(cleaned) < len(sample_dirty_qa_pairs)

    def test_removes_duplicates(self, sample_dirty_qa_pairs):
        """Test that duplicates are removed."""
        cleaned, stats = clean_qa_dataset(sample_dirty_qa_pairs, skip_duplicates=True)
        assert stats["duplicate"] > 0

    def test_stats_include_counts(self, sample_dirty_qa_pairs):
        """Test that stats include various counts."""
        cleaned, stats = clean_qa_dataset(sample_dirty_qa_pairs)
        assert "total" in stats
        assert "valid" in stats

    def test_relaxed_mode_keeps_more(self, sample_dirty_qa_pairs):
        """Test that relaxed mode keeps more pairs."""
        strict_cleaned, _ = clean_qa_dataset(sample_dirty_qa_pairs, strict_mode=True)
        relaxed_cleaned, _ = clean_qa_dataset(sample_dirty_qa_pairs, strict_mode=False)
        assert len(relaxed_cleaned) >= len(strict_cleaned)

    def test_keeps_conversational_data(self):
        """Test that conversational data is kept."""
        data = [
            {"question": "Hi", "answer": "Hello! How can I help you today?"},
            {"question": "Thanks", "answer": "You're welcome! Happy to help."},
        ]
        cleaned, stats = clean_qa_dataset(data, strict_mode=True)
        assert stats["conversational"] == 2

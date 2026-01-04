"""
Unit tests for paraphrase_generator.py
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.generators.paraphrase_generator import (
    extract_topic_from_question,
    get_question_type,
    generate_paraphrases,
    augment_qa_pair,
    augment_qa_dataset,
    QUESTION_PREFIXES,
    IMPERATIVE_TEMPLATES
)


class TestExtractTopic:
    """Tests for extract_topic_from_question function."""

    def test_what_is_question(self):
        """Test extracting topic from 'What is' question."""
        result = extract_topic_from_question("What is machine learning?")
        assert "machine learning" in result.lower()

    def test_how_does_question(self):
        """Test extracting topic from 'How does' question."""
        result = extract_topic_from_question("How does photosynthesis work?")
        assert "photosynthesis" in result.lower()

    def test_why_is_question(self):
        """Test extracting topic from 'Why is' question."""
        result = extract_topic_from_question("Why is the sky blue?")
        assert "sky blue" in result.lower()

    def test_removes_articles(self):
        """Test that articles are removed from topic."""
        result = extract_topic_from_question("What is a computer?")
        assert result.lower() == "computer"

    def test_handles_complex_questions(self):
        """Test handling complex questions."""
        result = extract_topic_from_question("Can you tell me what artificial intelligence is?")
        assert "artificial intelligence" in result.lower()


class TestGetQuestionType:
    """Tests for get_question_type function."""

    def test_what_question(self):
        """Test detecting 'what' question type."""
        assert get_question_type("What is Python?") == "what"

    def test_how_question(self):
        """Test detecting 'how' question type."""
        assert get_question_type("How does it work?") == "how"

    def test_why_question(self):
        """Test detecting 'why' question type."""
        assert get_question_type("Why is this important?") == "why"

    def test_when_question(self):
        """Test detecting 'when' question type."""
        assert get_question_type("When was it invented?") == "when"

    def test_where_question(self):
        """Test detecting 'where' question type."""
        assert get_question_type("Where is it located?") == "where"

    def test_who_question(self):
        """Test detecting 'who' question type."""
        assert get_question_type("Who invented the telephone?") == "who"

    def test_unknown_type(self):
        """Test handling unknown question type."""
        assert get_question_type("Tell me about Python") is None


class TestGenerateParaphrases:
    """Tests for generate_paraphrases function."""

    def test_includes_original(self):
        """Test that original question is always included."""
        question = "What is machine learning?"
        result = generate_paraphrases(question, num_variations=3)
        assert question in result

    def test_generates_variations(self):
        """Test that variations are generated."""
        question = "What is machine learning?"
        result = generate_paraphrases(question, num_variations=3)
        assert len(result) >= 2  # At least original + 1 variation

    def test_no_duplicates(self):
        """Test that no duplicates are in variations."""
        question = "What is machine learning?"
        result = generate_paraphrases(question, num_variations=5)
        assert len(result) == len(set(result))

    def test_includes_imperatives(self):
        """Test that imperative variations are included when requested."""
        question = "What is Python?"
        result = generate_paraphrases(question, num_variations=10, include_imperatives=True)
        # Should have at least one imperative style
        imperatives = [v for v in result if not v.endswith("?")]
        assert len(imperatives) >= 1

    def test_respects_num_variations(self):
        """Test that number of variations is respected."""
        question = "What is AI?"
        result = generate_paraphrases(question, num_variations=2)
        # Original + 2 variations = 3 max
        assert len(result) <= 3


class TestAugmentQAPair:
    """Tests for augment_qa_pair function."""

    def test_returns_list(self):
        """Test that function returns a list."""
        qa_pair = {"question": "What is Python?", "answer": "A programming language."}
        result = augment_qa_pair(qa_pair, num_variations=2)
        assert isinstance(result, list)

    def test_preserves_answer(self):
        """Test that answer is preserved across all variations."""
        qa_pair = {"question": "What is Python?", "answer": "A programming language."}
        result = augment_qa_pair(qa_pair, num_variations=2)
        for pair in result:
            assert pair["answer"] == "A programming language."

    def test_adds_original_question_field(self):
        """Test that original_question field is added."""
        qa_pair = {"question": "What is Python?", "answer": "A programming language."}
        result = augment_qa_pair(qa_pair, num_variations=2)
        for pair in result:
            assert pair["original_question"] == "What is Python?"

    def test_handles_empty_question(self):
        """Test handling of empty question."""
        qa_pair = {"question": "", "answer": "Some answer"}
        result = augment_qa_pair(qa_pair, num_variations=2)
        assert len(result) == 1  # Just returns original


class TestAugmentQADataset:
    """Tests for augment_qa_dataset function."""

    def test_augments_multiple_pairs(self, sample_qa_pairs):
        """Test augmenting multiple QA pairs."""
        result = augment_qa_dataset(sample_qa_pairs, variations_per_question=2)
        # Each pair should have at least original + variations
        assert len(result) >= len(sample_qa_pairs)

    def test_preserves_all_answers(self, sample_qa_pairs):
        """Test that all original answers are preserved."""
        result = augment_qa_dataset(sample_qa_pairs, variations_per_question=2)
        original_answers = {p["answer"] for p in sample_qa_pairs}
        result_answers = {p["answer"] for p in result}
        assert original_answers == result_answers


class TestConstants:
    """Tests for module constants."""

    def test_question_prefixes_exist(self):
        """Test that QUESTION_PREFIXES is defined."""
        assert QUESTION_PREFIXES is not None
        assert len(QUESTION_PREFIXES) > 0

    def test_imperative_templates_exist(self):
        """Test that IMPERATIVE_TEMPLATES is defined."""
        assert IMPERATIVE_TEMPLATES is not None
        assert len(IMPERATIVE_TEMPLATES) > 0

    def test_prefixes_have_variations(self):
        """Test that each question type has multiple prefixes."""
        for q_type, prefixes in QUESTION_PREFIXES.items():
            assert len(prefixes) >= 2, f"{q_type} should have multiple prefixes"

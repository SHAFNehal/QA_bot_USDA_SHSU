"""
Unit tests for conversational_data.py
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.generators.conversational_data import (
    GREETINGS,
    GRATITUDE,
    FAREWELLS,
    ACKNOWLEDGMENTS,
    CLARIFICATIONS,
    META_QUESTIONS,
    AFFIRMATIVES,
    NEGATIVES,
    get_conversational_qa_pairs,
    get_greetings,
    get_gratitude,
    get_farewells,
)


class TestGreetings:
    """Tests for greeting data."""

    def test_greetings_not_empty(self):
        """Test that greetings list is not empty."""
        assert len(GREETINGS) > 0

    def test_greetings_have_input_output(self):
        """Test that each greeting has input and output."""
        for greeting in GREETINGS:
            assert "input" in greeting
            assert "output" in greeting

    def test_greetings_inputs_are_strings(self):
        """Test that greeting inputs are strings."""
        for greeting in GREETINGS:
            assert isinstance(greeting["input"], str)
            assert len(greeting["input"]) > 0

    def test_greetings_outputs_are_friendly(self):
        """Test that greeting outputs are friendly responses."""
        for greeting in GREETINGS:
            output = greeting["output"].lower()
            # Should contain some friendly element
            friendly_words = ["hello", "hi", "hey", "good", "help", "assist", "welcome"]
            assert any(word in output for word in friendly_words), f"Output not friendly: {greeting['output']}"


class TestGratitude:
    """Tests for gratitude data."""

    def test_gratitude_not_empty(self):
        """Test that gratitude list is not empty."""
        assert len(GRATITUDE) > 0

    def test_gratitude_responses_are_appropriate(self):
        """Test that gratitude responses are appropriate."""
        for item in GRATITUDE:
            output = item["output"].lower()
            # Should contain acknowledgment
            ack_words = ["welcome", "glad", "happy", "help", "pleasure", "anytime"]
            assert any(word in output for word in ack_words), f"Response not appropriate: {item['output']}"


class TestFarewells:
    """Tests for farewell data."""

    def test_farewells_not_empty(self):
        """Test that farewells list is not empty."""
        assert len(FAREWELLS) > 0

    def test_farewells_are_polite(self):
        """Test that farewell responses are polite."""
        for item in FAREWELLS:
            output = item["output"].lower()
            # Should contain polite farewell
            farewell_words = ["bye", "goodbye", "care", "well", "soon", "day"]
            assert any(word in output for word in farewell_words), f"Farewell not polite: {item['output']}"


class TestAcknowledgments:
    """Tests for acknowledgment data."""

    def test_acknowledgments_not_empty(self):
        """Test that acknowledgments list is not empty."""
        assert len(ACKNOWLEDGMENTS) > 0

    def test_acknowledgments_have_valid_structure(self):
        """Test that acknowledgments have valid structure."""
        for item in ACKNOWLEDGMENTS:
            assert "input" in item
            assert "output" in item
            assert len(item["input"]) > 0
            assert len(item["output"]) > 0


class TestClarifications:
    """Tests for clarification data."""

    def test_clarifications_not_empty(self):
        """Test that clarifications list is not empty."""
        assert len(CLARIFICATIONS) > 0

    def test_clarifications_offer_help(self):
        """Test that clarification responses offer to help."""
        for item in CLARIFICATIONS:
            output = item["output"].lower()
            # Should offer to help clarify
            help_words = ["help", "clarify", "explain", "understand", "specific", "question"]
            assert any(word in output for word in help_words), f"Doesn't offer help: {item['output']}"


class TestMetaQuestions:
    """Tests for meta question data."""

    def test_meta_questions_not_empty(self):
        """Test that meta questions list is not empty."""
        assert len(META_QUESTIONS) > 0

    def test_meta_questions_describe_assistant(self):
        """Test that meta question responses describe the assistant."""
        for item in META_QUESTIONS:
            output = item["output"].lower()
            # Should describe capabilities
            desc_words = ["assistant", "help", "answer", "question", "can", "able"]
            assert any(word in output for word in desc_words), f"Doesn't describe assistant: {item['output']}"


class TestAffirmativesAndNegatives:
    """Tests for affirmative and negative responses."""

    def test_affirmatives_not_empty(self):
        """Test that affirmatives list is not empty."""
        assert len(AFFIRMATIVES) > 0

    def test_negatives_not_empty(self):
        """Test that negatives list is not empty."""
        assert len(NEGATIVES) > 0


class TestGetConversationalQAPairs:
    """Tests for get_conversational_qa_pairs function."""

    def test_returns_list(self):
        """Test that function returns a list."""
        result = get_conversational_qa_pairs()
        assert isinstance(result, list)

    def test_returns_qa_format(self):
        """Test that returned items are in QA format."""
        result = get_conversational_qa_pairs()
        for item in result:
            assert "question" in item
            assert "answer" in item

    def test_includes_all_categories(self):
        """Test that all categories are included."""
        result = get_conversational_qa_pairs()
        # Should have items from multiple categories
        expected_min = len(GREETINGS) + len(GRATITUDE) + len(FAREWELLS)
        assert len(result) >= expected_min

    def test_no_duplicates(self):
        """Test that there are no duplicate questions."""
        result = get_conversational_qa_pairs()
        questions = [item["question"] for item in result]
        assert len(questions) == len(set(questions))


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_greetings(self):
        """Test get_greetings function."""
        result = get_greetings()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_get_gratitude(self):
        """Test get_gratitude function."""
        result = get_gratitude()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_get_farewells(self):
        """Test get_farewells function."""
        result = get_farewells()
        assert isinstance(result, list)
        assert len(result) > 0

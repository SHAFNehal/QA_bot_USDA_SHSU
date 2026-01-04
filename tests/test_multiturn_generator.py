"""
Unit tests for multiturn_generator.py
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.generators.multiturn_generator import (
    FOLLOWUP_TEMPLATES,
    COREFERENCE_EXAMPLES,
    get_random_followup,
    format_conversation_for_training,
    generate_coreference_training_data,
    create_multiturn_from_qa,
    augment_dataset_with_coreference,
)


class TestFollowupTemplates:
    """Tests for FOLLOWUP_TEMPLATES constant."""

    def test_templates_exist(self):
        """Test that followup templates exist."""
        assert FOLLOWUP_TEMPLATES is not None
        assert len(FOLLOWUP_TEMPLATES) > 0

    def test_templates_have_categories(self):
        """Test that templates are organized by category."""
        expected_categories = ["clarification", "detail", "example", "pronoun", "continuation"]
        for category in expected_categories:
            assert category in FOLLOWUP_TEMPLATES, f"Missing category: {category}"

    def test_each_category_has_templates(self):
        """Test that each category has multiple templates."""
        for category, templates in FOLLOWUP_TEMPLATES.items():
            assert len(templates) >= 2, f"{category} should have multiple templates"


class TestCoreferenceExamples:
    """Tests for COREFERENCE_EXAMPLES constant."""

    def test_examples_exist(self):
        """Test that coreference examples exist."""
        assert COREFERENCE_EXAMPLES is not None
        assert len(COREFERENCE_EXAMPLES) > 0

    def test_examples_have_turns(self):
        """Test that each example has turns."""
        for example in COREFERENCE_EXAMPLES:
            assert "turns" in example
            assert len(example["turns"]) >= 2

    def test_turns_have_user_and_assistant(self):
        """Test that each turn has user and assistant."""
        for example in COREFERENCE_EXAMPLES:
            for turn in example["turns"]:
                assert "user" in turn
                assert "assistant" in turn

    def test_followup_questions_use_pronouns(self):
        """Test that followup questions use pronouns (it, its, etc.)."""
        pronoun_found = False
        for example in COREFERENCE_EXAMPLES:
            for turn in example["turns"][1:]:  # Skip first turn
                user_msg = turn["user"].lower()
                if any(p in user_msg for p in ["it", "its", "they", "their", "this", "that"]):
                    pronoun_found = True
                    break
            if pronoun_found:
                break
        assert pronoun_found, "No pronoun usage found in followup questions"


class TestGetRandomFollowup:
    """Tests for get_random_followup function."""

    def test_returns_string(self):
        """Test that function returns a string."""
        result = get_random_followup()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_from_category(self):
        """Test that function returns from specified category."""
        result = get_random_followup("clarification")
        assert result in FOLLOWUP_TEMPLATES["clarification"]

    def test_handles_invalid_category(self):
        """Test that function handles invalid category."""
        result = get_random_followup("invalid_category")
        # Should return something (from random category)
        assert isinstance(result, str)


class TestFormatConversationForTraining:
    """Tests for format_conversation_for_training function."""

    def test_returns_list(self, sample_multiturn_conversation):
        """Test that function returns a list."""
        result = format_conversation_for_training(sample_multiturn_conversation)
        assert isinstance(result, list)

    def test_returns_correct_count(self, sample_multiturn_conversation):
        """Test that function returns one example per turn."""
        result = format_conversation_for_training(sample_multiturn_conversation)
        assert len(result) == len(sample_multiturn_conversation)

    def test_examples_have_required_fields(self, sample_multiturn_conversation):
        """Test that examples have required fields."""
        result = format_conversation_for_training(sample_multiturn_conversation)
        for example in result:
            assert "input" in example
            assert "output" in example
            assert "turn_number" in example
            assert "total_turns" in example

    def test_includes_history_in_later_turns(self, sample_multiturn_conversation):
        """Test that later turns include conversation history."""
        result = format_conversation_for_training(sample_multiturn_conversation)
        # Second turn should include first turn's content
        second_turn = result[1]
        assert sample_multiturn_conversation[0]["user"] in second_turn["input"]
        assert sample_multiturn_conversation[0]["assistant"] in second_turn["input"]

    def test_uses_correct_format(self, sample_multiturn_conversation):
        """Test that output uses correct TinyLlama format."""
        result = format_conversation_for_training(sample_multiturn_conversation)
        for example in result:
            assert "<|user|>" in example["input"]
            assert "<|assistant|>" in example["input"]


class TestGenerateCoreferenceTrainingData:
    """Tests for generate_coreference_training_data function."""

    def test_returns_list(self):
        """Test that function returns a list."""
        result = generate_coreference_training_data()
        assert isinstance(result, list)

    def test_returns_non_empty(self):
        """Test that function returns non-empty list."""
        result = generate_coreference_training_data()
        assert len(result) > 0

    def test_examples_have_correct_format(self):
        """Test that examples have correct format."""
        result = generate_coreference_training_data()
        for example in result:
            assert "input" in example
            assert "output" in example


class TestCreateMultiturnFromQA:
    """Tests for create_multiturn_from_qa function."""

    def test_returns_list(self):
        """Test that function returns a list."""
        qa_pair = {"question": "What is AI?", "answer": "AI is artificial intelligence."}
        result = create_multiturn_from_qa(qa_pair)
        assert isinstance(result, list)

    def test_includes_original_qa(self):
        """Test that original QA is included."""
        qa_pair = {"question": "What is AI?", "answer": "AI is artificial intelligence."}
        result = create_multiturn_from_qa(qa_pair)
        assert result[0]["user"] == qa_pair["question"]
        assert result[0]["assistant"] == qa_pair["answer"]

    def test_adds_followups(self):
        """Test that followup turns are added."""
        qa_pair = {"question": "What is AI?", "answer": "AI is artificial intelligence."}
        result = create_multiturn_from_qa(qa_pair, num_followups=2)
        assert len(result) == 3  # Original + 2 followups


class TestAugmentDatasetWithCoreference:
    """Tests for augment_dataset_with_coreference function."""

    def test_returns_list(self, sample_qa_pairs):
        """Test that function returns a list."""
        result = augment_dataset_with_coreference(sample_qa_pairs)
        assert isinstance(result, list)

    def test_includes_original_qa(self, sample_qa_pairs):
        """Test that original QA pairs are included."""
        result = augment_dataset_with_coreference(sample_qa_pairs, include_predefined=False)
        assert len(result) >= len(sample_qa_pairs)

    def test_includes_coreference_examples(self, sample_qa_pairs):
        """Test that predefined coreference examples are included."""
        result_with = augment_dataset_with_coreference(sample_qa_pairs, include_predefined=True)
        result_without = augment_dataset_with_coreference(sample_qa_pairs, include_predefined=False)
        assert len(result_with) > len(result_without)

    def test_examples_have_type_field(self, sample_qa_pairs):
        """Test that examples have type field."""
        result = augment_dataset_with_coreference(sample_qa_pairs)
        for example in result:
            assert "type" in example or "turn_number" in example

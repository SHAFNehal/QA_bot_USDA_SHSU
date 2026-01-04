"""
Model Evaluation Script

This script evaluates a fine-tuned model on various test cases including:
- Greeting responses
- Rephrased questions
- Follow-up questions (multi-turn)
- Domain-specific QA accuracy

Usage:
    python evaluate.py --model_path fine_tuned_weights
    python evaluate.py --model_path fine_tuned_weights --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

import argparse
import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class EvaluationResult:
    """Result of a single evaluation test."""
    category: str
    test_name: str
    input_text: str
    expected_keywords: List[str]
    actual_response: str
    passed: bool
    score: float
    notes: str = ""


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    model_path: str
    timestamp: str
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    category_scores: Dict[str, float] = field(default_factory=dict)
    results: List[EvaluationResult] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100

    def to_dict(self) -> Dict:
        return {
            "model_path": self.model_path,
            "timestamp": self.timestamp,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "overall_score": f"{self.overall_score:.1f}%",
            "category_scores": {k: f"{v:.1f}%" for k, v in self.category_scores.items()},
            "results": [
                {
                    "category": r.category,
                    "test_name": r.test_name,
                    "input": r.input_text,
                    "passed": r.passed,
                    "score": r.score,
                    "response": r.actual_response[:200] + "..." if len(r.actual_response) > 200 else r.actual_response,
                }
                for r in self.results
            ]
        }


# Test cases for evaluation
GREETING_TESTS = [
    {"input": "Hi", "expected_keywords": ["hello", "hi", "help", "assist"]},
    {"input": "Hello", "expected_keywords": ["hello", "hi", "help", "welcome"]},
    {"input": "Hey there", "expected_keywords": ["hello", "hi", "hey", "help"]},
    {"input": "Good morning", "expected_keywords": ["good", "morning", "help", "hello"]},
    {"input": "Good afternoon", "expected_keywords": ["good", "afternoon", "help", "hello"]},
]

GRATITUDE_TESTS = [
    {"input": "Thanks", "expected_keywords": ["welcome", "glad", "help", "happy"]},
    {"input": "Thank you so much", "expected_keywords": ["welcome", "glad", "pleasure", "happy"]},
    {"input": "I appreciate it", "expected_keywords": ["welcome", "glad", "help", "appreciate"]},
]

FAREWELL_TESTS = [
    {"input": "Bye", "expected_keywords": ["bye", "goodbye", "care", "well"]},
    {"input": "Goodbye", "expected_keywords": ["bye", "goodbye", "care", "well"]},
    {"input": "See you later", "expected_keywords": ["bye", "see", "later", "care"]},
]

# Rephrased question tests - same concept, different phrasing
REPHRASED_QUESTION_TESTS = [
    {
        "concept": "machine_learning",
        "variations": [
            "What is machine learning?",
            "Can you explain machine learning?",
            "Tell me about machine learning",
            "How would you define machine learning?",
            "Describe machine learning",
        ],
        "expected_keywords": ["learn", "data", "algorithm", "pattern", "train", "model", "ai", "artificial"]
    },
    {
        "concept": "photosynthesis",
        "variations": [
            "What is photosynthesis?",
            "How does photosynthesis work?",
            "Explain photosynthesis",
            "Can you describe photosynthesis?",
            "Tell me about photosynthesis",
        ],
        "expected_keywords": ["plant", "light", "sun", "oxygen", "carbon", "glucose", "energy", "chlorophyll"]
    },
]

# Multi-turn conversation tests
MULTITURN_TESTS = [
    {
        "name": "coreference_it",
        "turns": [
            {"user": "What is Python?", "check_keywords": ["programming", "language"]},
            {"user": "Why is it popular?", "check_keywords": ["popular", "easy", "simple", "library", "libraries"]},
        ]
    },
    {
        "name": "coreference_this",
        "turns": [
            {"user": "What is artificial intelligence?", "check_keywords": ["ai", "artificial", "intelligence", "machine"]},
            {"user": "What are some applications of this?", "check_keywords": ["application", "use", "example"]},
        ]
    },
]


def check_response_quality(response: str, expected_keywords: List[str]) -> tuple:
    """
    Check if response contains expected keywords.
    Returns (passed, score, matched_keywords).
    """
    response_lower = response.lower()
    matched = [kw for kw in expected_keywords if kw.lower() in response_lower]
    score = len(matched) / len(expected_keywords) if expected_keywords else 0
    passed = score >= 0.3  # At least 30% of keywords should match

    # Also check for bad responses
    bad_indicators = [
        "i don't know",
        "i cannot",
        "error",
        "exception",
        "sorry, i",
        "as an ai",
    ]
    for indicator in bad_indicators:
        if indicator in response_lower:
            passed = False
            score = max(0, score - 0.2)
            break

    return passed, score, matched


class ModelEvaluator:
    """Evaluator for fine-tuned QA models."""

    def __init__(self, model_path: str, base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_path = model_path
        self.base_model = base_model
        self.inference = None
        self.report = EvaluationReport(
            model_path=model_path,
            timestamp=datetime.now().isoformat()
        )

    def load_model(self):
        """Load the model for inference."""
        try:
            from inference import QAInference
            self.inference = QAInference(
                base_model_name=self.base_model,
                peft_model_path=self.model_path,
                max_history_turns=5
            )
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Running in dry-run mode (no actual model inference)")
            return False

    def generate_response(self, question: str, use_history: bool = False) -> str:
        """Generate response for a question."""
        if self.inference is None:
            return "[DRY RUN - No model loaded]"
        try:
            return self.inference.generate_answer(question, use_history=use_history)
        except Exception as e:
            return f"[ERROR: {e}]"

    def clear_history(self):
        """Clear conversation history."""
        if self.inference:
            self.inference.clear_history()

    def run_greeting_tests(self) -> List[EvaluationResult]:
        """Run greeting tests."""
        results = []
        for test in GREETING_TESTS:
            response = self.generate_response(test["input"])
            passed, score, matched = check_response_quality(response, test["expected_keywords"])

            result = EvaluationResult(
                category="greetings",
                test_name=f"greeting_{test['input'].lower().replace(' ', '_')}",
                input_text=test["input"],
                expected_keywords=test["expected_keywords"],
                actual_response=response,
                passed=passed,
                score=score,
                notes=f"Matched: {matched}"
            )
            results.append(result)
        return results

    def run_gratitude_tests(self) -> List[EvaluationResult]:
        """Run gratitude tests."""
        results = []
        for test in GRATITUDE_TESTS:
            response = self.generate_response(test["input"])
            passed, score, matched = check_response_quality(response, test["expected_keywords"])

            result = EvaluationResult(
                category="gratitude",
                test_name=f"gratitude_{test['input'].lower().replace(' ', '_')[:20]}",
                input_text=test["input"],
                expected_keywords=test["expected_keywords"],
                actual_response=response,
                passed=passed,
                score=score,
                notes=f"Matched: {matched}"
            )
            results.append(result)
        return results

    def run_farewell_tests(self) -> List[EvaluationResult]:
        """Run farewell tests."""
        results = []
        for test in FAREWELL_TESTS:
            response = self.generate_response(test["input"])
            passed, score, matched = check_response_quality(response, test["expected_keywords"])

            result = EvaluationResult(
                category="farewells",
                test_name=f"farewell_{test['input'].lower().replace(' ', '_')}",
                input_text=test["input"],
                expected_keywords=test["expected_keywords"],
                actual_response=response,
                passed=passed,
                score=score,
                notes=f"Matched: {matched}"
            )
            results.append(result)
        return results

    def run_rephrased_question_tests(self) -> List[EvaluationResult]:
        """Run rephrased question tests."""
        results = []
        for concept_test in REPHRASED_QUESTION_TESTS:
            concept = concept_test["concept"]
            expected_keywords = concept_test["expected_keywords"]

            for i, variation in enumerate(concept_test["variations"]):
                response = self.generate_response(variation)
                passed, score, matched = check_response_quality(response, expected_keywords)

                result = EvaluationResult(
                    category="rephrased_questions",
                    test_name=f"{concept}_variation_{i+1}",
                    input_text=variation,
                    expected_keywords=expected_keywords,
                    actual_response=response,
                    passed=passed,
                    score=score,
                    notes=f"Matched: {matched}"
                )
                results.append(result)
        return results

    def run_multiturn_tests(self) -> List[EvaluationResult]:
        """Run multi-turn conversation tests."""
        results = []
        for test in MULTITURN_TESTS:
            self.clear_history()
            test_name = test["name"]

            for i, turn in enumerate(test["turns"]):
                response = self.generate_response(turn["user"], use_history=True)
                passed, score, matched = check_response_quality(response, turn["check_keywords"])

                result = EvaluationResult(
                    category="multiturn",
                    test_name=f"{test_name}_turn_{i+1}",
                    input_text=turn["user"],
                    expected_keywords=turn["check_keywords"],
                    actual_response=response,
                    passed=passed,
                    score=score,
                    notes=f"Turn {i+1}, Matched: {matched}"
                )
                results.append(result)
        return results

    def run_all_tests(self) -> EvaluationReport:
        """Run all evaluation tests."""
        print("\n" + "=" * 60)
        print("RUNNING MODEL EVALUATION")
        print("=" * 60)

        # Run all test categories
        all_results = []

        print("\n[1/5] Running greeting tests...")
        all_results.extend(self.run_greeting_tests())

        print("[2/5] Running gratitude tests...")
        all_results.extend(self.run_gratitude_tests())

        print("[3/5] Running farewell tests...")
        all_results.extend(self.run_farewell_tests())

        print("[4/5] Running rephrased question tests...")
        all_results.extend(self.run_rephrased_question_tests())

        print("[5/5] Running multi-turn tests...")
        all_results.extend(self.run_multiturn_tests())

        # Compile report
        self.report.results = all_results
        self.report.total_tests = len(all_results)
        self.report.passed_tests = sum(1 for r in all_results if r.passed)
        self.report.failed_tests = self.report.total_tests - self.report.passed_tests

        # Calculate category scores
        categories = set(r.category for r in all_results)
        for category in categories:
            cat_results = [r for r in all_results if r.category == category]
            cat_passed = sum(1 for r in cat_results if r.passed)
            self.report.category_scores[category] = (cat_passed / len(cat_results)) * 100

        return self.report

    def print_report(self):
        """Print evaluation report to console."""
        report = self.report

        print("\n" + "=" * 60)
        print("EVALUATION REPORT")
        print("=" * 60)

        print(f"\nModel: {report.model_path}")
        print(f"Timestamp: {report.timestamp}")
        print(f"\nOverall Score: {report.overall_score:.1f}%")
        print(f"Tests Passed: {report.passed_tests}/{report.total_tests}")

        print("\nCategory Scores:")
        for category, score in report.category_scores.items():
            status = "PASS" if score >= 80 else "NEEDS IMPROVEMENT" if score >= 50 else "FAIL"
            print(f"  {category}: {score:.1f}% [{status}]")

        print("\nFailed Tests:")
        failed = [r for r in report.results if not r.passed]
        if failed:
            for r in failed[:10]:  # Show first 10 failures
                print(f"  - [{r.category}] {r.test_name}")
                print(f"    Input: {r.input_text}")
                print(f"    Response: {r.actual_response[:100]}...")
        else:
            print("  None!")

        print("\n" + "=" * 60)

        # Success criteria check
        print("\nSUCCESS CRITERIA CHECK:")
        criteria = [
            ("Greetings work", report.category_scores.get("greetings", 0) >= 80),
            ("Rephrased questions work", report.category_scores.get("rephrased_questions", 0) >= 60),
            ("Multi-turn works", report.category_scores.get("multiturn", 0) >= 60),
            ("Overall score >= 70%", report.overall_score >= 70),
        ]
        for name, passed in criteria:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {name}")

        print("=" * 60)

    def save_report(self, output_path: str):
        """Save report to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.report.to_dict(), f, indent=2)
        print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned QA model")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model weights"
    )

    parser.add_argument(
        "--base_model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Base model name"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for evaluation report (JSON)"
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run without loading model (for testing the evaluation framework)"
    )

    args = parser.parse_args()

    # Create evaluator
    evaluator = ModelEvaluator(args.model_path, args.base_model)

    # Load model unless dry run
    if not args.dry_run:
        if not evaluator.load_model():
            print("Warning: Could not load model, running in dry-run mode")

    # Run evaluation
    report = evaluator.run_all_tests()

    # Print report
    evaluator.print_report()

    # Save report if output path specified
    if args.output:
        evaluator.save_report(args.output)
    else:
        # Default output path
        output_path = os.path.join(
            os.path.dirname(args.model_path) if os.path.dirname(args.model_path) else ".",
            "evaluation_report.json"
        )
        evaluator.save_report(output_path)


if __name__ == "__main__":
    main()

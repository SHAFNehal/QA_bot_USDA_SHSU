"""
Model Evaluation Script

This script evaluates a fine-tuned model on various test cases from your data:
- Holdout set evaluation with full metrics (BLEU, ROUGE, etc.)
- Single-turn question tests (consistency across questions)
- Multi-turn conversation tests (context handling)

All tests are dynamically generated from the test data file.

Metrics:
- Keyword match (pass/fail, score) for all tests.
- Standard metrics on holdout tests: BLEU, ROUGE-1/2/L,
  embedding similarity (cosine), exact match, token F1. Optional: LLM-as-judge.

Usage:
    python evaluate.py --model_path fine_tuned_weights --test_data training_test.jsonl
    python evaluate.py --model_path fine_tuned_weights --test_data training_test.jsonl --test_data_only
    python evaluate.py --model_path fine_tuned_weights --test_data training_test.jsonl --no_embedding
    python evaluate.py --model_path fine_tuned_weights --test_data training_test.jsonl --llm_judge
"""

import argparse
import json
import os
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from src.utils.file_processor import load_jsonl
from src.inference.metrics import compute_all_metrics, aggregate_metrics


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
    reference_answer: Optional[str] = None  # when set, standard metrics (BLEU, ROUGE, etc.) are computed
    extra_metrics: Optional[Dict[str, float]] = None  # BLEU, ROUGE-1/2/L, embedding_sim, exact_match, token_f1, llm_judge


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
    standard_metrics_agg: Dict[str, float] = field(default_factory=dict)  # mean BLEU, ROUGE, etc. over tests with reference

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
            "standard_metrics_agg": self.standard_metrics_agg,
            "results": [
                {
                    "category": r.category,
                    "test_name": r.test_name,
                    "input": r.input_text,
                    "passed": r.passed,
                    "score": r.score,
                    "response": r.actual_response[:200] + "..." if len(r.actual_response) > 200 else r.actual_response,
                    **({"extra_metrics": r.extra_metrics} if r.extra_metrics else {}),
                }
                for r in self.results
            ]
        }


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


def _keywords_from_answer(answer: str, max_keywords: int = 15, min_word_len: int = 4) -> List[str]:
    """Extract significant words from ground-truth answer for holdout scoring."""
    words = re.findall(r'\b[a-zA-Z]+\b', answer.lower())
    # Prefer longer, substantive words
    keywords = list(dict.fromkeys(w for w in words if len(w) >= min_word_len))[:max_keywords]
    return keywords if keywords else words[:max_keywords]


class ModelEvaluator:
    """Evaluator for fine-tuned QA models."""

    def __init__(
        self,
        model_path: str,
        base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        use_embedding: bool = True,
        use_llm_judge: bool = False,
        max_new_tokens: int = 2048,
    ):
        self.model_path = model_path
        self.base_model = base_model
        self.use_embedding = use_embedding
        self.use_llm_judge = use_llm_judge
        self.max_new_tokens = max_new_tokens
        self.inference = None
        self.report = EvaluationReport(
            model_path=model_path,
            timestamp=datetime.now().isoformat()
        )

    def load_model(self):
        """Load the model for inference."""
        try:
            from src.inference.inference import QAInference
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
            return self.inference.generate_answer(
                question, use_history=use_history, max_new_tokens=self.max_new_tokens
            )
        except Exception as e:
            return f"[ERROR: {e}]"

    def clear_history(self):
        """Clear conversation history."""
        if self.inference:
            self.inference.clear_history()

    def run_rephrased_question_tests(self, test_data_path: str) -> List[EvaluationResult]:
        """
        Run rephrased question tests using actual data.
        Tests single-turn questions from the dataset.
        """
        results = []
        items = load_jsonl(test_data_path)
        if not items:
            print(f"Warning: No examples in test file {test_data_path}")
            return results

        # Filter single-turn questions only (not multi-turn conversations)
        single_turn_items = []
        for item in items:
            if "question" in item and "answer" in item:
                # Simple Q&A format - single turn
                single_turn_items.append(item)
            elif "input" in item and "output" in item:
                # Check if it's a single-turn conversation
                raw_input = item["input"]
                user_turns = re.findall(r"<\|user\|>\s*\n(.*?)</s>", raw_input, re.DOTALL)
                if len(user_turns) == 1:
                    # Single turn conversation
                    single_turn_items.append(item)

        # Limit to 10 tests for rephrased questions
        test_items = single_turn_items[:10] if len(single_turn_items) > 10 else single_turn_items

        for i, item in enumerate(test_items):
            self.clear_history()
            
            if "question" in item and "answer" in item:
                question = item["question"].strip()
                expected_answer = item["answer"].strip()
            elif "input" in item and "output" in item:
                raw_input = item["input"]
                expected_answer = item["output"].strip()
                user_turns = re.findall(r"<\|user\|>\s*\n(.*?)</s>", raw_input, re.DOTALL)
                question = user_turns[0].strip() if user_turns else ""
            else:
                continue

            if not question:
                continue

            # Generate response
            response = self.generate_response(question, use_history=False)
            
            # Extract keywords from expected answer
            expected_keywords = _keywords_from_answer(expected_answer)
            passed, score, matched = check_response_quality(response, expected_keywords)

            result = EvaluationResult(
                category="rephrased_questions",
                test_name=f"single_turn_q{i+1}",
                input_text=question,
                expected_keywords=expected_keywords,
                actual_response=response,
                passed=passed,
                score=score,
                notes=f"Matched: {matched}",
                reference_answer=expected_answer
            )
            results.append(result)
        
        return results

    def run_multiturn_tests(self, test_data_path: str) -> List[EvaluationResult]:
        """
        Run multi-turn conversation tests using actual data.
        Tests multi-turn conversations from the dataset.
        """
        results = []
        items = load_jsonl(test_data_path)
        if not items:
            print(f"Warning: No examples in test file {test_data_path}")
            return results

        # Filter multi-turn conversations only
        multiturn_items = []
        for item in items:
            if "input" in item and "output" in item:
                raw_input = item["input"]
                user_turns = re.findall(r"<\|user\|>\s*\n(.*?)</s>", raw_input, re.DOTALL)
                if len(user_turns) > 1:
                    # Multi-turn conversation
                    multiturn_items.append(item)

        # Limit to 4 tests for multi-turn
        test_items = multiturn_items[:4] if len(multiturn_items) > 4 else multiturn_items

        for i, item in enumerate(test_items):
            self.clear_history()
            
            raw_input = item["input"]
            expected_answer = item["output"].strip()
            user_turns = re.findall(r"<\|user\|>\s*\n(.*?)</s>", raw_input, re.DOTALL)
            user_turns = [t.strip() for t in user_turns if t.strip()]
            
            if len(user_turns) < 2:
                continue

            # Replay conversation history
            for turn_idx, turn in enumerate(user_turns[:-1]):
                self.generate_response(turn, use_history=True)
            
            # Generate response for the final turn
            final_question = user_turns[-1]
            response = self.generate_response(final_question, use_history=True)
            
            # Extract keywords from expected answer
            expected_keywords = _keywords_from_answer(expected_answer)
            passed, score, matched = check_response_quality(response, expected_keywords)

            result = EvaluationResult(
                category="multiturn",
                test_name=f"multiturn_conv{i+1}",
                input_text=f"[Turn {len(user_turns)}] {final_question}",
                expected_keywords=expected_keywords,
                actual_response=response,
                passed=passed,
                score=score,
                notes=f"Turns: {len(user_turns)}, Matched: {matched}",
                reference_answer=expected_answer
            )
            results.append(result)
        
        return results

    def run_holdout_tests(self, test_data_path: str) -> List[EvaluationResult]:
        """Run evaluation on the holdout test set (saved at training time)."""
        results = []
        items = load_jsonl(test_data_path)
        if not items:
            print(f"Warning: No examples in holdout file {test_data_path}")
            return results

        total = len(items)
        for i, item in enumerate(items):
            if (i + 1) % 100 == 0 or i == 0:
                print(f"  Holdout progress: {i+1}/{total}", flush=True)
            self.clear_history()
            input_text = ""
            if "question" in item and "answer" in item:
                question = item["question"].strip()
                expected_answer = item["answer"].strip()
                input_text = question
                response = self.generate_response(question, use_history=False)
            elif "input" in item and "output" in item:
                # Multi-turn: replay conversation then score last response
                raw_input = item["input"]
                expected_answer = item["output"].strip()
                user_turns = re.findall(r"<\|user\|>\s*\n(.*?)</s>", raw_input, re.DOTALL)
                user_turns = [t.strip() for t in user_turns if t.strip()]
                if not user_turns:
                    continue
                input_text = user_turns[-1]
                for turn in user_turns[:-1]:
                    self.generate_response(turn, use_history=True)
                response = self.generate_response(user_turns[-1], use_history=True)
            else:
                continue

            expected_keywords = _keywords_from_answer(expected_answer)
            passed, score, matched = check_response_quality(response, expected_keywords)

            # Standard metrics (BLEU, ROUGE, embedding sim, exact match, token F1, optional LLM judge)
            inference_fn = (lambda p: self.generate_response(p, use_history=False)) if self.use_llm_judge else None
            extra_metrics = compute_all_metrics(
                reference=expected_answer,
                candidate=response,
                question=input_text,
                inference_fn=inference_fn,
                use_embedding=self.use_embedding,
                use_llm_judge=self.use_llm_judge,
            )

            results.append(
                EvaluationResult(
                    category="holdout",
                    test_name=f"holdout_{i+1}",
                    input_text=input_text,
                    expected_keywords=expected_keywords,
                    actual_response=response,
                    passed=passed,
                    score=score,
                    notes=f"Matched: {matched}",
                    reference_answer=expected_answer,
                    extra_metrics=extra_metrics if extra_metrics else None,
                )
            )

        return results

    def run_all_tests(
        self,
        test_data_path: Optional[str] = None,
        test_data_only: bool = False,
    ) -> EvaluationReport:
        """Run evaluation tests. If test_data_only is True, run only on test_data_path (no preset tests)."""
        print("\n" + "=" * 60)
        print("RUNNING MODEL EVALUATION")
        print("=" * 60)

        all_results = []

        if not test_data_path or not os.path.isfile(test_data_path):
            print(f"\nError: test_data path is required. Got: {test_data_path}")
            print("All evaluation tests now require data from the test file.")
            self.report.results = []
            self.report.total_tests = 0
            self.report.passed_tests = 0
            self.report.failed_tests = 0
            return self.report

        if test_data_only:
            print("\n[1/1] Running evaluation on provided test data only (holdout metrics)...")
            all_results.extend(self.run_holdout_tests(test_data_path))
        else:
            # Run all three test categories from data
            print(f"\n[1/3] Running holdout (test set) evaluation with full metrics...")
            all_results.extend(self.run_holdout_tests(test_data_path))
            
            print(f"\n[2/3] Running single-turn question tests from data...")
            all_results.extend(self.run_rephrased_question_tests(test_data_path))
            
            print(f"[3/3] Running multi-turn conversation tests from data...")
            all_results.extend(self.run_multiturn_tests(test_data_path))

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

        # Aggregate standard metrics (BLEU, ROUGE, etc.) over results that have reference + extra_metrics
        results_with_metrics = [r for r in all_results if getattr(r, "extra_metrics", None)]
        if results_with_metrics:
            self.report.standard_metrics_agg = aggregate_metrics(results_with_metrics)

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

        if report.standard_metrics_agg:
            print("\nStandard metrics (on tests with reference, e.g. holdout):")
            for name, value in sorted(report.standard_metrics_agg.items()):
                print(f"  {name}: {value:.4f}")

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

        # Success criteria check (only for categories that were run)
        print("\nSUCCESS CRITERIA CHECK:")
        criteria = []
        thresholds = {
            "rephrased_questions": (60, "Single-turn questions"),
            "multiturn": (60, "Multi-turn conversations"),
            "holdout": (50, "Holdout with full metrics"),
        }
        for cat, score in report.category_scores.items():
            thresh, label = thresholds.get(cat, (50, f"{cat} score"))
            criteria.append((label, score >= thresh))
        criteria.append(("Overall score >= 70%", report.overall_score >= 70))
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

    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Path to test dataset (JSONL) - REQUIRED. All evaluation tests are generated from this data."
    )

    parser.add_argument(
        "--test_data_only",
        action="store_true",
        help="Evaluate only holdout tests with full metrics (skip single-turn and multi-turn category tests)."
    )

    parser.add_argument(
        "--no_embedding",
        action="store_true",
        help="Disable embedding/semantic similarity metric (avoids loading sentence-transformers)."
    )

    parser.add_argument(
        "--llm_judge",
        action="store_true",
        help="Enable LLM-as-judge: use the same model to rate each response 0-1 (slower)."
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Max tokens for generation (default: 2048)"
    )

    args = parser.parse_args()

    # Create evaluator
    evaluator = ModelEvaluator(
        args.model_path,
        args.base_model,
        use_embedding=not args.no_embedding,
        use_llm_judge=args.llm_judge,
        max_new_tokens=args.max_new_tokens,
    )

    # Load model unless dry run
    if not args.dry_run:
        if not evaluator.load_model():
            print("Warning: Could not load model, running in dry-run mode")

    # Run evaluation (optionally only on test_data, or full suite with optional holdout)
    report = evaluator.run_all_tests(
        test_data_path=args.test_data,
        test_data_only=args.test_data_only,
    )

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

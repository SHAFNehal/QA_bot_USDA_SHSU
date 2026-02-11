"""
Hybrid RAG System Evaluation Script

Evaluates the Hybrid RAG system (Fine-tuned + RAG) on:
- Answer quality (BLEU, ROUGE, embedding similarity, etc.)
- Response time
- Component analysis (fine-tuned vs RAG contribution)

Usage:
    python evaluate_hybrid_rag.py --test_data data_output/training_test.jsonl --model_path fine_tuned_weights
    python evaluate_hybrid_rag.py --test_data data_output/training_test.jsonl --model_path fine_tuned_weights --db_path rag_db_hybrid
"""

import argparse
import json
import os
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from src.utils.file_processor import load_jsonl
from src.inference.metrics import compute_all_metrics
from src.Hybrid_RAG.pipeline import HybridRAGPipeline


@dataclass
class HybridRAGEvaluationResult:
    """Result of a single Hybrid RAG evaluation test."""
    test_name: str
    question: str
    expected_answer: str
    generated_answer: str
    fine_tuned_answer: str
    rag_answer: str
    response_time: float
    passed: bool
    keyword_score: float
    matched_keywords: List[str]
    answer_metrics: Optional[Dict[str, float]] = None


@dataclass
class HybridRAGEvaluationReport:
    """Complete Hybrid RAG evaluation report."""
    model_path: str
    db_path: str
    timestamp: str
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    avg_response_time: float = 0.0
    answer_metrics: Dict[str, float] = field(default_factory=dict)
    results: List[HybridRAGEvaluationResult] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100

    def to_dict(self) -> Dict:
        return {
            "model_path": self.model_path,
            "db_path": self.db_path,
            "timestamp": self.timestamp,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "overall_score": f"{self.overall_score:.1f}%",
            "avg_response_time": f"{self.avg_response_time:.2f}s",
            "answer_metrics": self.answer_metrics,
            "results": [
                {
                    "test_name": r.test_name,
                    "question": r.question,
                    "passed": r.passed,
                    "keyword_score": r.keyword_score,
                    "response_time": f"{r.response_time:.2f}s",
                    "answer": r.generated_answer[:200] + "..." if len(r.generated_answer) > 200 else r.generated_answer,
                    **({"answer_metrics": r.answer_metrics} if r.answer_metrics else {}),
                }
                for r in self.results
            ]
        }


def extract_keywords(text: str, min_word_len: int = 4, max_keywords: int = 15) -> List[str]:
    """Extract significant keywords from text."""
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    keywords = list(dict.fromkeys(w for w in words if len(w) >= min_word_len))[:max_keywords]
    return keywords if keywords else words[:max_keywords]


def check_keyword_match(response: str, expected_keywords: List[str]) -> Tuple[bool, float, List[str]]:
    """
    Check if response contains expected keywords.
    Returns (passed, score, matched_keywords).
    """
    response_lower = response.lower()
    matched = [kw for kw in expected_keywords if kw.lower() in response_lower]
    score = len(matched) / len(expected_keywords) if expected_keywords else 0
    passed = score >= 0.3  # At least 30% of keywords should match

    # Check for bad responses
    bad_indicators = ["i don't know", "i cannot", "error", "exception", "sorry, i", "no information"]
    for indicator in bad_indicators:
        if indicator in response_lower:
            passed = False
            score = max(0, score - 0.2)
            break

    return passed, score, matched


class HybridRAGEvaluator:
    """Evaluator for Hybrid RAG systems."""

    def __init__(
        self,
        model_path: str,
        db_path: str = "rag_db_hybrid",
        base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        use_embedding: bool = True,
    ):
        self.model_path = model_path
        self.db_path = db_path
        self.base_model = base_model
        self.use_embedding = use_embedding
        
        self.report = HybridRAGEvaluationReport(
            model_path=model_path,
            db_path=db_path,
            timestamp=datetime.now().isoformat()
        )
        
        # Initialize Hybrid RAG pipeline
        print(f"Loading Hybrid RAG system...")
        print(f"  Model: {model_path}")
        print(f"  Database: {db_path}")
        
        self.hybrid_pipeline = HybridRAGPipeline(
            peft_model_path=model_path,
            base_model_name=base_model,
            db_path=db_path
        )
        print("✓ Hybrid RAG system loaded")

    def evaluate_single_query(
        self,
        question: str,
        expected_answer: str,
        test_name: str
    ) -> HybridRAGEvaluationResult:
        """Evaluate a single query."""
        
        # Measure response time
        start_time = time.time()
        
        # Query Hybrid RAG system
        result = self.hybrid_pipeline.query(question)
        
        response_time = time.time() - start_time
        
        # Extract responses
        generated_answer = result.get("final_answer", "")
        fine_tuned_answer = result.get("fine_tuned_answer", "")
        rag_answer = result.get("rag_answer", "")
        
        # Check keyword matching
        expected_keywords = extract_keywords(expected_answer)
        passed, keyword_score, matched = check_keyword_match(generated_answer, expected_keywords)
        
        # Compute standard NLP metrics
        answer_metrics = None
        if self.use_embedding:
            answer_metrics = compute_all_metrics(
                reference=expected_answer,
                candidate=generated_answer,
                question=question,
                inference_fn=None,
                use_embedding=True,
                use_llm_judge=False,
            )
        
        return HybridRAGEvaluationResult(
            test_name=test_name,
            question=question,
            expected_answer=expected_answer,
            generated_answer=generated_answer,
            fine_tuned_answer=fine_tuned_answer,
            rag_answer=rag_answer,
            response_time=response_time,
            passed=passed,
            keyword_score=keyword_score,
            matched_keywords=matched,
            answer_metrics=answer_metrics,
        )

    def run_evaluation(self, test_data_path: str) -> HybridRAGEvaluationReport:
        """Run evaluation on test dataset."""
        print("\n" + "=" * 60)
        print("RUNNING HYBRID RAG EVALUATION")
        print("=" * 60)
        
        # Load test data
        items = load_jsonl(test_data_path)
        if not items:
            print(f"Error: No test data found in {test_data_path}")
            return self.report
        
        print(f"\nEvaluating {len(items)} test cases...")
        
        results = []
        total_time = 0.0
        
        for i, item in enumerate(items, 1):
            # Extract question and answer
            if "question" in item and "answer" in item:
                question = item["question"].strip()
                expected_answer = item["answer"].strip()
            elif "input" in item and "output" in item:
                # Extract last user turn from conversation
                raw_input = item["input"]
                expected_answer = item["output"].strip()
                user_turns = re.findall(r"<\|user\|>\s*\n(.*?)</s>", raw_input, re.DOTALL)
                question = user_turns[-1].strip() if user_turns else ""
            else:
                continue
            
            if not question:
                continue
            
            # Evaluate
            print(f"  [{i}/{len(items)}] Evaluating: {question[:60]}...")
            try:
                result = self.evaluate_single_query(
                    question=question,
                    expected_answer=expected_answer,
                    test_name=f"test_{i}"
                )
                results.append(result)
                total_time += result.response_time
            except Exception as e:
                print(f"    ⚠️  Error: {str(e)}")
                continue
        
        # Compile report
        self.report.results = results
        self.report.total_tests = len(results)
        self.report.passed_tests = sum(1 for r in results if r.passed)
        self.report.failed_tests = self.report.total_tests - self.report.passed_tests
        self.report.avg_response_time = total_time / len(results) if results else 0
        
        # Aggregate answer metrics (BLEU, ROUGE, etc.)
        results_with_metrics = [r for r in results if r.answer_metrics]
        if results_with_metrics:
            agg_metrics = {}
            metric_names = results_with_metrics[0].answer_metrics.keys()
            for metric_name in metric_names:
                values = [r.answer_metrics[metric_name] for r in results_with_metrics if metric_name in r.answer_metrics]
                if values:
                    agg_metrics[metric_name] = sum(values) / len(values)
            self.report.answer_metrics = agg_metrics
        
        return self.report

    def print_report(self):
        """Print evaluation report to console."""
        report = self.report

        print("\n" + "=" * 60)
        print("HYBRID RAG EVALUATION REPORT")
        print("=" * 60)

        print(f"\nModel: {report.model_path}")
        print(f"Database: {report.db_path}")
        print(f"Timestamp: {report.timestamp}")
        print(f"\nOverall Score: {report.overall_score:.1f}%")
        print(f"Tests Passed: {report.passed_tests}/{report.total_tests}")
        print(f"Avg Response Time: {report.avg_response_time:.2f}s")

        if report.answer_metrics:
            print("\nAnswer Quality Metrics:")
            for name, value in sorted(report.answer_metrics.items()):
                print(f"  {name}: {value:.4f}")

        print("\nFailed Tests:")
        failed = [r for r in report.results if not r.passed]
        if failed:
            for r in failed[:10]:  # Show first 10 failures
                print(f"  - {r.test_name}")
                print(f"    Q: {r.question[:80]}...")
                print(f"    A: {r.generated_answer[:80]}...")
                print(f"    Score: {r.keyword_score:.2f}")
        else:
            print("  None!")

        print("\n" + "=" * 60)
        
        # Success criteria
        print("\nSUCCESS CRITERIA CHECK:")
        criteria = [
            ("Overall score >= 70%", report.overall_score >= 70),
            ("Avg response time < 8s", report.avg_response_time < 8.0),
        ]
        
        if report.answer_metrics:
            if "bleu" in report.answer_metrics:
                criteria.append(("BLEU >= 0.3", report.answer_metrics["bleu"] >= 0.3))
            if "embedding_similarity" in report.answer_metrics:
                criteria.append(("Embedding similarity >= 0.6", report.answer_metrics["embedding_similarity"] >= 0.6))
        
        for name, passed in criteria:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {name}")
        
        print("=" * 60)

    def save_report(self, output_path: str):
        """Save report to JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.report.to_dict(), f, indent=2)
        print(f"\n✓ Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Hybrid RAG system")

    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Path to test dataset (JSONL)"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model weights"
    )

    parser.add_argument(
        "--db_path",
        type=str,
        default="rag_db_hybrid",
        help="Path to RAG database"
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
        "--no_embedding",
        action="store_true",
        help="Disable embedding similarity metric"
    )

    args = parser.parse_args()

    # Set default output path
    if args.output is None:
        test_dir = os.path.dirname(args.test_data)
        args.output = os.path.join(test_dir, "hybrid_rag_evaluation_report.json")

    # Run evaluation
    evaluator = HybridRAGEvaluator(
        model_path=args.model_path,
        db_path=args.db_path,
        base_model=args.base_model,
        use_embedding=not args.no_embedding,
    )

    evaluator.run_evaluation(args.test_data)
    evaluator.print_report()
    evaluator.save_report(args.output)


if __name__ == "__main__":
    main()

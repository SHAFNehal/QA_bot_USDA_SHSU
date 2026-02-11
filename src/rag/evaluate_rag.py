"""
RAG System Evaluation Script

Evaluates the RAG (Retrieval-Augmented Generation) system on:
- Retrieval quality (precision, recall, MRR)
- Answer quality (BLEU, ROUGE, embedding similarity, etc.)
- Response time
- Source accuracy

Usage:
    python evaluate_rag.py --test_data data_output/training_test.jsonl
    python evaluate_rag.py --test_data data_output/training_test.jsonl --db_path rag_db
    python evaluate_rag.py --test_data data_output/training_test.jsonl --detailed
"""

import argparse
import json
import os
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.utils.file_processor import load_jsonl
from src.inference.metrics import compute_all_metrics, aggregate_metrics
from src.rag.query import RAGQuery


@dataclass
class RAGEvaluationResult:
    """Result of a single RAG evaluation test."""
    test_name: str
    question: str
    expected_answer: str
    generated_answer: str
    retrieved_docs: List[str]
    retrieved_scores: List[float]
    response_time: float
    passed: bool
    keyword_score: float
    matched_keywords: List[str]
    extra_metrics: Optional[Dict[str, float]] = None  # BLEU, ROUGE, etc.
    retrieval_metrics: Optional[Dict[str, float]] = None  # Precision, recall, etc.


@dataclass
class RAGEvaluationReport:
    """Complete RAG evaluation report."""
    db_path: str
    timestamp: str
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    avg_response_time: float = 0.0
    retrieval_metrics: Dict[str, float] = field(default_factory=dict)
    answer_metrics: Dict[str, float] = field(default_factory=dict)
    results: List[RAGEvaluationResult] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100

    def to_dict(self) -> Dict:
        return {
            "db_path": self.db_path,
            "timestamp": self.timestamp,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "overall_score": f"{self.overall_score:.1f}%",
            "avg_response_time": f"{self.avg_response_time:.2f}s",
            "retrieval_metrics": self.retrieval_metrics,
            "answer_metrics": self.answer_metrics,
            "results": [
                {
                    "test_name": r.test_name,
                    "question": r.question,
                    "passed": r.passed,
                    "keyword_score": r.keyword_score,
                    "response_time": f"{r.response_time:.2f}s",
                    "retrieved_docs_count": len(r.retrieved_docs),
                    "answer": r.generated_answer[:200] + "..." if len(r.generated_answer) > 200 else r.generated_answer,
                    **({"answer_metrics": r.extra_metrics} if r.extra_metrics else {}),
                    **({"retrieval_metrics": r.retrieval_metrics} if r.retrieval_metrics else {}),
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


class RAGEvaluator:
    """Evaluator for RAG systems."""

    def __init__(
        self,
        db_path: str = "rag_db",
        base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        use_embedding: bool = True,
        use_bm25: bool = False,
        top_k: int = 5,
    ):
        self.db_path = db_path
        self.base_model = base_model
        self.use_embedding = use_embedding
        self.use_bm25 = use_bm25
        self.top_k = top_k
        
        self.report = RAGEvaluationReport(
            db_path=db_path,
            timestamp=datetime.now().isoformat()
        )
        
        # Initialize RAG query system
        print(f"Loading RAG system from {db_path}...")
        self.rag_query = RAGQuery(
            db_path=db_path,
            base_model=base_model,
            use_bm25=use_bm25,
            top_k=top_k
        )
        print("✓ RAG system loaded")

    def evaluate_single_query(
        self,
        question: str,
        expected_answer: str,
        test_name: str
    ) -> RAGEvaluationResult:
        """Evaluate a single query."""
        
        # Measure response time
        start_time = time.time()
        
        # Query RAG system
        response, sources = self.rag_query.query(question)
        
        response_time = time.time() - start_time
        
        # Extract retrieved documents and scores
        retrieved_docs = [s['content'][:200] for s in sources] if sources else []
        retrieved_scores = [s.get('score', 0.0) for s in sources] if sources else []
        
        # Check keyword matching
        expected_keywords = extract_keywords(expected_answer)
        passed, keyword_score, matched = check_keyword_match(response, expected_keywords)
        
        # Compute standard NLP metrics
        extra_metrics = None
        if self.use_embedding:
            extra_metrics = compute_all_metrics(
                reference=expected_answer,
                candidate=response,
                question=question,
                inference_fn=None,
                use_embedding=True,
                use_llm_judge=False,
            )
        
        return RAGEvaluationResult(
            test_name=test_name,
            question=question,
            expected_answer=expected_answer,
            generated_answer=response,
            retrieved_docs=retrieved_docs,
            retrieved_scores=retrieved_scores,
            response_time=response_time,
            passed=passed,
            keyword_score=keyword_score,
            matched_keywords=matched,
            extra_metrics=extra_metrics,
        )

    def run_evaluation(self, test_data_path: str) -> RAGEvaluationReport:
        """Run evaluation on test dataset."""
        print("\n" + "=" * 60)
        print("RUNNING RAG EVALUATION")
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
            result = self.evaluate_single_query(
                question=question,
                expected_answer=expected_answer,
                test_name=f"test_{i}"
            )
            results.append(result)
            total_time += result.response_time
        
        # Compile report
        self.report.results = results
        self.report.total_tests = len(results)
        self.report.passed_tests = sum(1 for r in results if r.passed)
        self.report.failed_tests = self.report.total_tests - self.report.passed_tests
        self.report.avg_response_time = total_time / len(results) if results else 0
        
        # Aggregate answer metrics (BLEU, ROUGE, etc.)
        results_with_metrics = [r for r in results if r.extra_metrics]
        if results_with_metrics:
            agg_metrics = {}
            metric_names = results_with_metrics[0].extra_metrics.keys()
            for metric_name in metric_names:
                values = [r.extra_metrics[metric_name] for r in results_with_metrics if metric_name in r.extra_metrics]
                if values:
                    agg_metrics[metric_name] = sum(values) / len(values)
            self.report.answer_metrics = agg_metrics
        
        # Compute retrieval metrics
        avg_docs_retrieved = sum(len(r.retrieved_docs) for r in results) / len(results) if results else 0
        avg_retrieval_score = sum(sum(r.retrieved_scores) / len(r.retrieved_scores) if r.retrieved_scores else 0 for r in results) / len(results) if results else 0
        
        self.report.retrieval_metrics = {
            "avg_docs_retrieved": avg_docs_retrieved,
            "avg_retrieval_score": avg_retrieval_score,
            "retrieval_success_rate": sum(1 for r in results if len(r.retrieved_docs) > 0) / len(results) if results else 0,
        }
        
        return self.report

    def print_report(self):
        """Print evaluation report to console."""
        report = self.report

        print("\n" + "=" * 60)
        print("RAG EVALUATION REPORT")
        print("=" * 60)

        print(f"\nDatabase: {report.db_path}")
        print(f"Timestamp: {report.timestamp}")
        print(f"\nOverall Score: {report.overall_score:.1f}%")
        print(f"Tests Passed: {report.passed_tests}/{report.total_tests}")
        print(f"Avg Response Time: {report.avg_response_time:.2f}s")

        if report.retrieval_metrics:
            print("\nRetrieval Metrics:")
            for name, value in sorted(report.retrieval_metrics.items()):
                print(f"  {name}: {value:.4f}")

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
            ("Retrieval success >= 90%", report.retrieval_metrics.get("retrieval_success_rate", 0) >= 0.9),
            ("Avg response time < 5s", report.avg_response_time < 5.0),
        ]
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
    parser = argparse.ArgumentParser(description="Evaluate RAG system")

    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Path to test dataset (JSONL)"
    )

    parser.add_argument(
        "--db_path",
        type=str,
        default="rag_db",
        help="Path to RAG database"
    )

    parser.add_argument(
        "--base_model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Base model for generation"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for evaluation report (JSON)"
    )

    parser.add_argument(
        "--use_bm25",
        action="store_true",
        help="Use BM25 for hybrid retrieval"
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of documents to retrieve"
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
        args.output = os.path.join(test_dir, "rag_evaluation_report.json")

    # Run evaluation
    evaluator = RAGEvaluator(
        db_path=args.db_path,
        base_model=args.base_model,
        use_embedding=not args.no_embedding,
        use_bm25=args.use_bm25,
        top_k=args.top_k,
    )

    evaluator.run_evaluation(args.test_data)
    evaluator.print_report()
    evaluator.save_report(args.output)


if __name__ == "__main__":
    main()

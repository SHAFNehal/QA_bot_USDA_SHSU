"""
Hybrid RAG query CLI: load pipeline, run interactive or single question.
"""

import argparse
import sys

from src.Hybrid_RAG.pipeline import HybridRAGPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Query the Hybrid RAG pipeline (restate -> RAG + FT -> synthesize)."
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="rag_db",
        help="Path to Chroma RAG DB",
    )
    parser.add_argument(
        "--peft-model",
        type=str,
        required=True,
        help="Path to fine-tuned model weights (required)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Base model name for the fine-tuned model",
    )
    parser.add_argument(
        "--small-llm",
        type=str,
        default=None,
        help="Small LLM for restate/synthesize (default from config)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Read questions from stdin in a loop",
    )
    parser.add_argument(
        "question",
        type=str,
        nargs="*",
        help="Question to answer (ignored in interactive mode)",
    )
    args = parser.parse_args()

    pipeline = HybridRAGPipeline(
        db_path=args.db_path,
        peft_model_path=args.peft_model,
        base_model_name=args.base_model,
        small_llm_name=args.small_llm,
    )

    if args.interactive:
        print("Hybrid RAG (interactive). Enter a question and press Enter. Empty line or Ctrl-D to quit.")
        while True:
            try:
                q = input("Question: ").strip()
            except EOFError:
                break
            if not q:
                continue
            print("Answer:", pipeline.answer(q))
            print()
    else:
        q = " ".join(args.question).strip() if args.question else None
        if not q:
            print("Provide a question as argument or use --interactive.", file=sys.stderr)
            sys.exit(1)
        print(pipeline.answer(q))


if __name__ == "__main__":
    main()

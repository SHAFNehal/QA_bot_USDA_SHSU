"""
RAG query CLI: load pipeline from db_path, run interactive loop or single question.
"""

import argparse
import sys

from src.rag.pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Query the RAG pipeline: interactive or single question."
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="rag_db",
        help="Path to Chroma DB (and optional BM25 index)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for generator (default from config)",
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
        help="Question(s) to answer (single string if one arg; ignored in interactive mode)",
    )
    args = parser.parse_args()

    pipeline = RAGPipeline(db_path=args.db_path, model_name=args.model)

    if args.interactive:
        print("RAG query (interactive). Enter a question and press Enter. Empty line or Ctrl-D to quit.")
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

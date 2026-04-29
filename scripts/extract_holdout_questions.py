"""Extract unique questions and optional holdout sample from qa_cleaned. Run from project root."""
import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QA_FILE = os.path.join(PROJECT_ROOT, "data_output", "qa_cleaned.jsonl")
OUT_QUESTIONS = os.path.join(PROJECT_ROOT, "data_output", "holdout_questions_only.txt")
OUT_SAMPLE = os.path.join(PROJECT_ROOT, "data_output", "holdout_from_training_sample.jsonl")

def main():
    seen = set()
    questions = []
    with open(QA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                q = o.get("original_question") or o.get("question", "")
                q = q.strip()
                if q and q not in seen:
                    seen.add(q)
                    questions.append(q)
            except Exception:
                pass

    os.makedirs(os.path.dirname(OUT_QUESTIONS), exist_ok=True)
    with open(OUT_QUESTIONS, "w", encoding="utf-8") as f:
        for q in questions[:200]:
            f.write(q + "\n")
    print(f"holdout_questions_only.txt: {min(200, len(questions))} questions")

    seen2 = set()
    pairs = []
    with open(QA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if len(pairs) >= 30:
                break
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                q = (o.get("original_question") or o.get("question", "")).strip()
                a = (o.get("answer") or "").strip()
                if q and a and q not in seen2:
                    seen2.add(q)
                    pairs.append({"question": q, "answer": a})
            except Exception:
                pass

    with open(OUT_SAMPLE, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"holdout_from_training_sample.jsonl: {len(pairs)} pairs (FROM TRAINING - debug only)")


if __name__ == "__main__":
    main()

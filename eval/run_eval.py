"""Run the evaluation questions and save the results.

Usage:  python -m eval.run_eval

A question counts as correct when:
  - answerable: the answer contains all expected keywords
  - unanswerable: the system returned "insufficient information"
Retrieval is counted as a hit when one of the expected (file, page) sources
is among the retrieved chunks.
I still read through the results by hand afterwards (see eval/results.md).
"""
import csv
import json
import os
import time
import unicodedata

from app import rag, vector_store
from app.pdf_processor import process_pdf

DATA_DIR = "data"


def main():
    # index the sample documents if they are not in the vector DB yet
    indexed = vector_store.list_files()
    for name in sorted(os.listdir(DATA_DIR)):
        if name.endswith(".pdf") and name not in indexed:
            with open(os.path.join(DATA_DIR, name), "rb") as f:
                vector_store.add_chunks(process_pdf(f.read(), name))

    with open("eval/questions.json") as f:
        questions = json.load(f)

    rows = []
    for q in questions:
        result = rag.answer_question(q["question"])
        retrieved = {(h["file_name"], h["page"]) for h in vector_store.search(q["question"])}
        expected = {tuple(s) for s in q["expected_sources"]}

        if not expected:
            correct = not result["answered"]
            retrieval_hit = "-"
        else:
            # the model sometimes uses special spaces/hyphens, e.g. "July\u202f2024", "pre‑deployment"
            answer = unicodedata.normalize("NFKC", result["answer"]).lower()
            for dash in ["‑", "‐", "–", "-"]:
                answer = answer.replace(dash, " ")
            correct = result["answered"] and all(k.lower() in answer for k in q["keywords"])
            retrieval_hit = "Y" if expected & retrieved else "N"

        sources = sorted({(c["file_name"], c["page"]) for c in result["citations"]})
        rows.append({
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "retrieved_answer": result["answer"].replace("\n", " "),
            "source": "; ".join(f"{f} p{p}" for f, p in sources) or "-",
            "expected_source": "; ".join(f"{f} p{p}" for f, p in sorted(expected)) or "-",
            "retrieval_hit": retrieval_hit,
            "correct": "Y" if correct else "N",
        })
        print(f"[{rows[-1]['correct']}] {q['question']}")
        time.sleep(1)  # stay under the Groq free-tier rate limit

    with open("eval/results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    correct = sum(r["correct"] == "Y" for r in rows)
    hits = [r for r in rows if r["retrieval_hit"] != "-"]
    hit_count = sum(r["retrieval_hit"] == "Y" for r in hits)
    print(f"\nAnswer accuracy: {correct}/{len(rows)}")
    print(f"Retrieval hit rate (top-k contains an expected source): {hit_count}/{len(hits)}")


if __name__ == "__main__":
    main()

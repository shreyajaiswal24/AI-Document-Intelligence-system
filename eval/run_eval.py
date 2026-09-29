"""Ask all evaluation questions and save the answers to eval/results.csv.

Usage:  python -m eval.run_eval

After running it, I compared each answer with the expected answer and the
PDF page, and filled in the "correct" column (Y/N) by hand.
"""
import csv
import json
import os

from app import rag, vector_store
from app.pdf_processor import process_pdf


def load_documents():
    already_loaded = vector_store.list_files()
    for name in os.listdir("data"):
        if name.endswith(".pdf") and name not in already_loaded:
            with open(os.path.join("data", name), "rb") as f:
                vector_store.add_chunks(process_pdf(f.read(), name))


def main():
    load_documents()

    with open("eval/questions.json") as f:
        questions = json.load(f)

    rows = []
    for q in questions:
        result = rag.answer_question(q["question"])

        sources = []
        for c in result["citations"]:
            source = f"{c['file_name']} p{c['page']}"
            if source not in sources:
                sources.append(source)

        rows.append({
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "retrieved_answer": result["answer"].replace("\n", " "),
            "source": ", ".join(sources) or "-",
            "expected_source": q["expected_source"],
            "correct": "",  # filled in by hand after reading the answer
        })
        print("Q:", q["question"])
        print("A:", result["answer"][:200], "\n")

    with open("eval/results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("Saved eval/results.csv")


if __name__ == "__main__":
    main()

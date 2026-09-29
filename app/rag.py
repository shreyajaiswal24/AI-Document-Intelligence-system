import logging
import re

from app import llm, vector_store
from app.config import MAX_DISTANCE

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    pass


def answer_question(question):
    try:
        hits = vector_store.search(question)
    except Exception as e:
        logger.error("Vector DB search failed: %s", e)
        raise VectorStoreError("Vector database error, please try again")

    # drop chunks that are not similar enough to the question
    relevant = [h for h in hits if h["distance"] <= MAX_DISTANCE]

    if not relevant:
        # nothing close enough -> don't even ask the LLM, so it can't make things up
        logger.info("No relevant chunks for question, returning no-answer")
        return {"answer": llm.NO_ANSWER, "citations": [], "answered": False}

    relevant = vector_store.add_next_chunks(relevant)
    answer = llm.generate_answer(question, relevant)
    # gpt-oss sometimes writes citations as 【1】 or [1†L4-L6] instead of [1]
    answer = answer.replace("【", "[").replace("】", "]")
    answer = re.sub(r"\[(\d+)†[^\]]*\]", r"[\1]", answer)

    if llm.NO_ANSWER.lower() in answer.lower():
        return {"answer": llm.NO_ANSWER, "citations": [], "answered": False}

    # keep only the chunks the model actually cited, e.g. [1], [3]
    cited_numbers = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
    citations = []
    for n in cited_numbers:
        if 1 <= n <= len(relevant):
            c = relevant[n - 1]
            citations.append({
                "ref": n,
                "file_name": c["file_name"],
                "page": c["page"],
                "chunk_id": c["chunk_id"],
                "distance": c["distance"],
                "text": c["text"][:300],
            })

    return {"answer": answer, "citations": citations, "answered": True}

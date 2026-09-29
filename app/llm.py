import logging
import time

from groq import Groq

from app.config import FALLBACK_MODEL, GROQ_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)

NO_ANSWER = "Insufficient information in the provided documents to answer this question."

SYSTEM_PROMPT = f"""You answer questions using ONLY the context given below.

Rules:
- Use only facts from the context. Do not use outside knowledge.
- After each fact, cite the source like [1] or [2] using the context numbers.
- If the context does not contain the answer, reply exactly: "{NO_ANSWER}"
- Keep the answer short and clear."""


class LLMError(Exception):
    pass


def build_context(chunks):
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[{i}] (file: {c['file_name']}, page {c['page']})\n{c['text']}")
    return "\n\n".join(parts)


def generate_answer(question, chunks):
    if not GROQ_API_KEY:
        raise LLMError("GROQ_API_KEY is not set")

    client = Groq(api_key=GROQ_API_KEY, timeout=30)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{build_context(chunks)}\n\nQuestion: {question}"},
    ]

    # try the main model first, then the fallback model
    for model in [LLM_MODEL, FALLBACK_MODEL]:
        try:
            start = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=800,
            )
            answer = response.choices[0].message.content.strip()
            logger.info("LLM call ok: model=%s time=%.2fs tokens=%s",
                        model, time.time() - start, response.usage.total_tokens)
            return answer
        except Exception as e:
            logger.error("LLM call failed with model %s: %s", model, e)

    raise LLMError("LLM service is not available right now")

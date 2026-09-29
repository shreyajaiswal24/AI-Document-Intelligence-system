# Evaluation Results

**Documents** (both public domain, in `data/`):
- `NIST.AI.100-1.pdf` — AI Risk Management Framework 1.0 (48 pages, 277 chunks)
- `NIST.AI.600-1.pdf` — Generative AI Profile (64 pages, 401 chunks)

**Settings:** chunk size 600, overlap 100, top-k 5 (+ next chunk added as context), max cosine distance 0.75, model `openai/gpt-oss-120b` on Groq
**Run it:** `python -m eval.run_eval` (full answers are saved in `results.csv`)

| # | Question | Expected answer | System answer (short) | Source | Correct |
|---|----------|-----------------|-----------------------|--------|---------|
| 1 | What are the four functions of the AI RMF Core? | GOVERN, MAP, MEASURE, MANAGE | GOVERN, MAP, MEASURE, MANAGE | 100-1 p25 | Y |
| 2 | Which law directed NIST to develop the AI RMF? | National AI Initiative Act of 2020 | Insufficient information | - | **N** |
| 3 | Which AI RMF function is described as cross-cutting? | GOVERN | GOVERN, infused throughout the other three | 100-1 p25 | Y |
| 4 | Why are AI systems described as socio-technical? | Influenced by societal dynamics and human behavior | Same | 100-1 p6 | Y |
| 5 | Characteristics of trustworthy AI? | Valid & reliable, safe, secure & resilient, accountable & transparent, explainable & interpretable, privacy-enhanced, fair | All 7 | 100-1 p17 | Y |
| 6 | Where can comments on the AI RMF Playbook be sent? | AIframework@nist.gov | AIframework@nist.gov | 100-1 p3 | Y |
| 7 | How does the GAI Profile define confabulation? | Confidently stated but false content | Same, also mentions "hallucinations" | 600-1 p10 | Y |
| 8 | What does the Environmental Impacts risk refer to? | High compute/energy use for training and running models | Energy- and carbon-intensive training | 600-1 p12 | Y |
| 9 | Four primary considerations of the GAI Public Working Group? | Governance, Content Provenance, Pre-deployment Testing, Incident Disclosure | Insufficient information | - | **N** |
| 10 | Which GAI risks fall under misuse by humans? | CBRN, Data Privacy, Human-AI Configuration, Obscene Content, Information Integrity, Information Security | All 6 **plus "Homogenization"** (wrong) | 600-1 p7 | **N** (manual) |
| 11 | When was the GAI Profile published? | July 2024 | July 2024 | 600-1 p2 | Y |
| 12 | Which executive order is the GAI Profile a response to? | EO 14110 | EO 14110 | 600-1 p5 | Y |
| 13 | Annual budget of the U.S. AI Safety Institute? | Not in documents | Insufficient information | - | Y |
| 14 | How many parameters does GPT-4 have? | Not in documents | Insufficient information | - | Y |
| 15 | Who won the 2022 FIFA World Cup? | Not in documents | Insufficient information | - | Y |

## Summary

| Metric | Result |
|---|---|
| Answer accuracy (automatic keyword check) | 13 / 15 |
| **Answer accuracy (after reading every answer myself)** | **12 / 15** |
| Retrieval hit rate (an expected page is in the top 5) | 10 / 12 |
| Correct abstentions on unanswerable questions | 3 / 3 |
| Hallucinated answers to unanswerable questions | 0 |

The automatic check says a question is correct if the answer contains all expected keywords (or abstains for unanswerable ones). It marked Q10 correct because all 6 expected risks were there, but reading the answer showed an extra wrong item, so I count it as wrong. Keyword checks can't catch extra wrong content — that's why I review manually.

**How the abstentions happened:** only Q15 (World Cup, best distance 0.85) was stopped by the distance threshold before calling the LLM. Q13 (0.39) and Q14 (0.73) passed the threshold — the AI Safety Institute and GPT-4 are mentioned in the documents — and the LLM correctly answered "insufficient information" because of the prompt rule. So both layers matter: the threshold only catches clearly off-topic questions.

## Failure analysis

**Q2 — retrieval miss (vocabulary mismatch).** The question says "law", the document says "National Artificial Intelligence Initiative Act ... As directed by". The top 5 chunks were other general pages about the AI RMF (all with distance ~0.2, very similar to each other), so the right chunk didn't make it. I tested keyword search (BM25) on this question and it didn't find it either — "law" isn't in the text. Fix: query rewriting (ask the LLM to rephrase the question before searching) or a stronger embedding model.

**Q9 — retrieval miss.** The answer is on p6 of the GAI Profile, but the small embedding model ranked other "working group / public" pages higher. **BM25 ranked the correct page #1** for this question, so hybrid search (BM25 + vectors) would fix this one.

**Q10 — the model added a wrong item.** The source text lists the three risk groups one after another on the same line: `...Harmful Bias, and Homogenization; 2) Misuse by humans (or malicious use): CBRN...`. The model pulled "Homogenization" from the end of group 1 into group 2. This is an LLM reading error, not retrieval.

## What I changed during evaluation

1. **Ligatures.** These PDFs store "fi" as a single character (`ﬁ`), so words like "conﬁdently" didn't match "confidently". Added Unicode NFKC normalisation to the cleaner.
2. **Information split across chunks.** The first run scored Q10 with only 1 of 6 risks: the list started at the end of one chunk and continued in the next one, which wasn't retrieved. Now, for each retrieved chunk, the next chunk on the same page is added as context. This fixed Q10's list and Q3 (the "cross-cutting" sentence was in the following chunk). Together with fixes 3 and 4, automatic accuracy went from 9/15 to 13/15.
3. **Expected sources.** My first expected pages were too narrow: confabulation is defined on p8 *and* in its own section on p10; Environmental Impacts on p8 and p12. Both are valid, so both count now.
4. **Eval scoring.** The model sometimes writes "July 2024" with a special narrow space (` `) or uses non-standard hyphens, so the keyword check failed on correct answers. The answer is now normalised before comparing.

## Chunk size and top-k

Retrieval hit rate on the 12 answerable questions:

| chunk size / overlap | k=3 | k=5 | k=8 |
|---|---|---|---|
| 400 / 80 | 8/12 | 10/12 | 11/12 |
| 600 / 100 | 8/12 | 10/12 | 11/12 |
| 800 / 150 | 9/12 | 10/12 | 10/12 |
| 1000 / 200 | 8/12 | 9/12 | 11/12 |

Sizes from 400 to 800 perform about the same here; 1000 is worse at k=5. I kept 600/100 with k=5: it is well within the embedding model's 256-token limit, and k=5 plus next-chunk expansion gives the LLM enough context without a huge prompt. k=8 would gain one question but doubles the prompt size.

Note: the hit rate is measured by page, so it can be a bit optimistic — a hit means *a* chunk from the right page was retrieved, not necessarily the chunk with the answer (this happened in Q3 before the next-chunk fix).

## Possible improvements

- **Hybrid search (BM25 + vectors):** tested, fixes Q9.
- **Query rewriting** before retrieval: would help vocabulary mismatches like Q2.
- **Reranker** (cross-encoder) on the top 10-20 chunks.
- **Drop reference/bibliography pages** from the index: they show up in results for broad questions (e.g. p60-61 of the GAI Profile) without useful content.
- **Bigger evaluation set** and an LLM-as-judge to catch errors like Q10 automatically.

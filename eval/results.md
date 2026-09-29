# Evaluation Results

**Document:** `data/Dukaan-Saathi.pdf` — a 12-page hackathon pitch deck (20 chunks)

**Settings:** chunk size 600, overlap 100, top 5 chunks (+ next chunk added), max distance 0.75, `openai/gpt-oss-120b` on Groq

**How I evaluated:** I wrote 15 questions — 11 with answers in the deck and 4 that the deck can't answer. `python -m eval.run_eval` asks all of them and saves the answers to `results.csv`. Then I read each answer, compared it with the expected answer and the PDF page, and marked it correct (Y/N) by hand.

| # | Question | Expected answer | Retrieved answer (short) | Source | Correct |
|---|---|---|---|---|---|
| 1 | Which hackathon and track? | Paytm Build for India AI Hackathon, Track 1: Merchant Growth AI | Same | p1 | Y |
| 2 | Who is the merchant persona? | Ramesh, kirana + chai counter in Dadar, Mumbai | Same | p2 | Y |
| 3 | What does Regulars Radar do? | Finds repeat payers who went quiet, suggests a nudge | Same | p5 | Y |
| 4 | Which LLM is used, and the fallback? | Sarvam-M, Groq fallback | Same | p8 | Y |
| 5 | Which tool is used for memory? | Cognee (knowledge graph of the shop) | Same | p8 | Y |
| 6 | What are the steps of the agent loop? | Observe, Diagnose, Recommend, Approve, Act, Learn | Same | p7 | Y |
| 7 | Can Dukaan Saathi move money? | No — only reads payments and pre-fills forms | Same | p7 | Y |
| 8 | Yesterday's sales in the sample brief? | 6,200, 18% below Tuesday average | Same | p4 | Y |
| 9 | What happens at 1:45 in the demo? | One-tap offer approval, n8n workflow fires | Same | p10 | Y |
| 10 | Which n8n actions are in the MVP? | Offer, restock order, loan pre-fill | Same | p11 | Y |
| 11 | What does the team ask Paytm for? | Sandbox API access + one real merchant to pilot with | Same | p12 | Y |
| 12 | How many merchants in the pilot? | Not stated — the deck has a blank "[__] merchants" | **"One real merchant"** | p12 | **N** |
| 13 | Cost per month for a merchant? | Not in document | Insufficient information | - | Y |
| 14 | Did Dukaan Saathi win the hackathon? | Not in document | Insufficient information | - | Y |
| 15 | Who won the 2022 FIFA World Cup? | Not in document | Insufficient information | - | Y |

## Summary

- **Answer accuracy: 14 / 15**
- Answerable questions: 11 / 11 correct, all with the right page cited
- Unanswerable questions: 3 / 4 correctly refused

## Failure: Q12 (pilot size)

The deck mentions two different pilots:
- p11 (roadmap): "Pilot with **[__]** merchants in Mumbai" — the number was left blank.
- p12 (ask to Paytm): "one real merchant in Mumbai to pilot the morning brief with".

The model found the p12 sentence and answered "one real merchant", mixing up the two. It isn't a made-up answer (the text is really there), but it answers a different question. Possible fixes: tell the model in the prompt to treat blanks like "[__]" as missing information, or make sure both chunks are shown so it can notice the difference.

## Chunk size and top-k

How often the right page was in the top k results, for the 11 answerable questions:

| chunk size / overlap | k=3 | k=5 | k=8 |
|---|---|---|---|
| 400 / 80 | 10/11 | 10/11 | 10/11 |
| **600 / 100** | 10/11 | **11/11** | 11/11 |
| 800 / 150 | 9/11 | 9/11 | 10/11 |
| 1000 / 200 | 9/11 | 9/11 | 10/11 |

600/100 was the best (or tied) at every k. With k=5 every expected page is found, so I kept k=5 — k=8 wouldn't find anything more and would send 40% of this short document to the LLM for every question.

Note: I tuned these on the same 15 questions and one short document, so the numbers are optimistic. A longer document would probably need a higher k.

## Other observations

- Similarity distance of the best chunk was 0.33–0.68 for answerable questions and 0.95 for the World Cup question, so the 0.75 threshold stops clearly off-topic questions before calling the LLM.
- The other three unanswerable questions (0.45–0.58) passed the threshold because they are about Dukaan Saathi; the prompt rule is what made the model refuse. Both layers are needed.
- The deck is made of slides with short phrases and tables, and extraction still worked well; e.g. the agent loop (Q6) and the demo timeline (Q9) came out correctly even though they are laid out as diagrams/tables.

## Possible improvements

- **Prompt rule for blanks/placeholders** (fixes Q12).
- **Hybrid search (keywords + vectors)** for exact terms and names.
- **Table-aware extraction**, so each table row becomes its own chunk.
- **More documents and more questions**, and a second person marking the answers.

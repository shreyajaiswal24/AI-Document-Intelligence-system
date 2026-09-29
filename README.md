# AI Document Intelligence System

Upload PDF documents and ask questions about them. Answers come only from the documents, with citations (file + page). If the answer isn't in the documents, the system says so instead of guessing.

**Stack:** FastAPI · PyMuPDF · ChromaDB · all-MiniLM-L6-v2 embeddings · Groq (`openai/gpt-oss-120b`)

**Live demo:** https://ai-document-intelligence-system-nrqumvzxycdpvhcvygyfjq.streamlit.app/

## Architecture

```mermaid
flowchart LR
    subgraph Upload ["POST /upload"]
        A[PDF file] --> B[Validate<br/>type, size, empty]
        B --> C[PyMuPDF<br/>extract text per page]
        C --> D[Clean text]
        D --> E[Split into chunks<br/>600 chars, 100 overlap]
        E --> F[Embed<br/>all-MiniLM-L6-v2]
        F --> G[(ChromaDB<br/>text + file, page, chunk id)]
    end

    subgraph Ask ["POST /ask"]
        Q[Question] --> H[Embed question]
        H --> I[Top 5 similar chunks]
        G --> I
        I --> J{Any chunk with<br/>distance ≤ 0.75?}
        J -- no --> K[Insufficient information<br/>LLM not called]
        J -- yes --> X[Add next chunk<br/>from same page]
        X --> L[Prompt with numbered context]
        L --> M[Groq LLM<br/>fallback to smaller model]
        M --> N[Answer + citations]
    end
```

### Project structure

```
app/
  config.py         settings from environment variables
  pdf_processor.py  extract, clean and chunk PDF text
  vector_store.py   ChromaDB: store chunks and search
  llm.py            prompt + Groq call (with fallback model)
  rag.py            retrieve -> filter -> add next chunk -> generate -> citations
  main.py           FastAPI endpoints
streamlit_app.py    simple web UI (same pipeline, no HTTP)
eval/
  questions.json    15 evaluation questions
  run_eval.py       runs them and writes results.csv
  results.md        results and discussion
tests/              API and unit tests
data/               sample PDFs used for the evaluation
```

## Setup

```bash
git clone <repo-url> && cd doc-qa-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then add your GROQ_API_KEY
uvicorn app.main:app --reload
```

Open http://localhost:8000/docs for the Swagger UI.

The embedding model (~80 MB) is downloaded automatically the first time.

**Docker:**
```bash
docker build -t doc-qa .
docker run -p 8000:8000 --env-file .env doc-qa
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | **Required.** Groq API key |
| `LLM_MODEL` | `openai/gpt-oss-120b` | Main model |
| `FALLBACK_MODEL` | `openai/gpt-oss-20b` | Used if the main model fails |
| `CHROMA_DIR` | `chroma_db` | Where the vector DB is saved |
| `CHUNK_SIZE` | `600` | Chunk size in characters |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `TOP_K` | `5` | Number of chunks retrieved |
| `MAX_DISTANCE` | `0.75` | Chunks less similar than this are ignored |
| `MAX_FILE_MB` | `20` | Upload size limit |

## Web UI

A simple Streamlit page to upload PDFs and ask questions. Live version: https://ai-document-intelligence-system-nrqumvzxycdpvhcvygyfjq.streamlit.app/

To run it locally:

```bash
streamlit run streamlit_app.py
```

## API

### `POST /upload`
Upload a PDF (multipart form, field name `file`).

```bash
curl -F "file=@data/Dukaan-Saathi.pdf" http://localhost:8000/upload
```
```json
{"file_name": "Dukaan-Saathi.pdf", "pages_with_text": 12, "chunks": 20}
```

| Status | When |
|---|---|
| 400 | not a `.pdf`, empty file, or corrupted PDF |
| 413 | file too large |
| 422 | no text found (e.g. scanned PDF) |
| 503 | vector DB error |

Uploading the same file again replaces its chunks (no duplicates).

### `POST /ask`

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Which LLM does Dukaan Saathi use?"}'
```
Returns the answer and the sources it used (file name and page for each citation):
```json
{
  "answer": "Dukaan Saathi uses Sarvam-M as its main LLM, with a Groq model as the fallback [2].",
  "citations": [{"file_name": "Dukaan-Saathi.pdf", "page": 8}],
  "answered": true
}
```
If the answer is not in the documents, `answer` is "Insufficient information..." and `answered` is `false`.

| Status | When |
|---|---|
| 400 | empty question, or no documents uploaded yet |
| 503 | Groq API key missing, LLM unavailable, or vector DB error |
| 500 | unexpected error (logged) |

### `GET /health`
```json
{"status": "ok", "vector_db": "ok", "groq_api_key_set": true, "llm_model": "openai/gpt-oss-120b", "chunks_indexed": 20}
```
`status` is `degraded` if the vector DB is down or the API key is missing.

## Design decisions

### PDF processing
- **PyMuPDF** reads the text page by page, so every chunk knows which page it came from.
- Cleaning is kept light: fix special characters (like the "ﬁ" ligature), remove page-number lines, re-join hyphenated words and remove extra spaces.
- Each page is chunked separately, so every citation points to exactly one page.

### Chunking — 600 characters, 100 overlap
- The embedding model only reads about 1000 characters, so chunks must be smaller than that.
- I tested 400, 600, 800 and 1000 characters (see [eval/results.md](eval/results.md)). 600 was the best (or tied) at every top-k setting.
- Smaller chunks usually hold one topic, which makes search more accurate.
- Chunks are cut at a paragraph, sentence or word boundary, never in the middle of a word.
- The 100-character overlap means a sentence on the border between two chunks is not lost.

### Embeddings — all-MiniLM-L6-v2
- Runs locally on CPU: free, fast, and no extra API key.
- Good quality for English semantic search.
- Downside: it's a small model. It can struggle with exact codes/terms and with table chunks that mix several facts. A bigger model would do better.

### Vector store — ChromaDB
- Stores the text, embeddings and metadata (file, page, chunk ID) together.
- Saves to a local folder, so the project runs right after cloning — nothing else to install or host.
- For production I would use a hosted DB like Qdrant or Pinecone.

### Retrieval — top 5 chunks + similarity threshold
- **Top 5:** in my test the right page was in the top 3 for 10/11 questions and in the top 5 for 11/11. Going to 8 found nothing more and would send much more text to the LLM.
- **Next chunk added:** for each retrieved chunk, the next chunk on the same page is also sent to the LLM. Lists and definitions are often cut between two chunks, so this gives the LLM the full text.
- **Threshold (distance ≤ 0.75):** if no chunk is similar enough, the system answers "insufficient information" without calling the LLM at all. Real questions scored 0.33–0.68; an off-topic one ("Who won the 2022 FIFA World Cup?") scored 0.95.

### LLM and prompt
- **Groq + `openai/gpt-oss-120b`:** fast, low cost and follows instructions well. Temperature is 0 for consistent answers.
- The prompt says: use only the given context, cite every fact as `[1]`, `[2]`..., and if the answer isn't there, reply with an exact "insufficient information" sentence.
- Only the chunks the model actually cited are returned as sources.
- **Two layers against hallucination:** the threshold stops off-topic questions, and the prompt rule handles questions that sound related but aren't answered in the documents (e.g. "How much will Dukaan Saathi cost a merchant per month?" — the product is described, the price isn't).
- **Fallback:** if the main model fails (outage, rate limit, timeout), the smaller `gpt-oss-20b` is tried automatically.

### Logging
Uploads (pages, chunks), retrieval (similarity scores), LLM calls (model, time, tokens) and all errors are logged with timestamps.

## Evaluation

15 questions on `data/Dukaan-Saathi.pdf` (a hackathon pitch deck): 11 with answers in the deck and 4 that it can't answer. The script collects the answers; I then checked each one against the PDF by hand.

| Result | |
|---|---|
| **Answer accuracy** | **14 / 15** |
| Answerable questions correct (with the right page cited) | 11 / 11 |
| Unanswerable questions correctly refused | 3 / 4 |

The one failure: asked how many merchants will be in the pilot, the model answered "one real merchant". The deck actually leaves the number blank ("[__] merchants") and mentions "one real merchant" in a different sentence, so the model mixed the two up.

Full results and analysis: [eval/results.md](eval/results.md). Run it with `python -m eval.run_eval`.

## Tests

```bash
pytest -q
```
Covers file validation (wrong type, empty, fake or text-less PDF), empty questions, citation parsing, the "no LLM call when nothing is relevant" rule, vector-DB failure handling and text cleaning/splitting.

## Limitations

- No OCR — scanned PDFs are rejected.
- Tables and dense lists lose their layout when extracted as plain text.
- Search is vector-only, so exact names and terms are sometimes missed.
- Chunks don't cross pages, so a paragraph continuing on the next page is split.
- No conversation history — each question is independent.
- All documents share one collection (no per-user separation), and the API has no delete endpoint.
- The threshold and top-k were tuned on one short document and 15 questions, so longer or very different documents may need a higher top-k or a different threshold.
- Uploads are processed inside the request, so a very large PDF takes a while. (Large files are saved in batches, and only the top chunks go to the LLM, so size doesn't affect the prompt.)

## Scaling and production ideas

- **1,000 concurrent users:** several API workers behind a load balancer, a hosted vector DB shared by all of them, and uploads processed in a background queue.
- **Lower cost:** cache answers to repeated questions, send fewer but better chunks, and use the smaller model for simple questions.
- **Better retrieval:** hybrid search (keywords + vectors), a reranker, and rewriting the question before searching.
- **Monitoring:** track response time, token usage, errors and how often the system says "insufficient information".

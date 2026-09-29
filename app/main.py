import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from app import config, rag, vector_store
from app.llm import LLMError
from app.pdf_processor import process_pdf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("app")

app = FastAPI(title="AI Document Intelligence System")


class AskRequest(BaseModel):
    question: str


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    try:
        chunks = vector_store.count()
        db_status = "ok"
    except Exception as e:
        logger.error("Vector DB health check failed: %s", e)
        chunks = 0
        db_status = "error"

    return {
        "status": "ok" if db_status == "ok" and config.GROQ_API_KEY else "degraded",
        "vector_db": db_status,
        "groq_api_key_set": bool(config.GROQ_API_KEY),
        "llm_model": config.LLM_MODEL,
        "chunks_indexed": chunks,
    }


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    logger.info("Upload received: %s", file.filename)

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    data = await file.read()
    if len(data) == 0:
        raise HTTPException(400, "The file is empty")
    if len(data) > config.MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {config.MAX_FILE_MB} MB")
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, "The file is not a valid PDF")

    try:
        chunks = process_pdf(data, file.filename)
    except ValueError as e:
        logger.warning("Could not read %s: %s", file.filename, e)
        raise HTTPException(400, str(e))

    if not chunks:
        # usually a scanned PDF (images only), we don't do OCR
        raise HTTPException(422, "No text could be extracted from this PDF (it may be a scanned document)")

    try:
        vector_store.add_chunks(chunks)
    except Exception as e:
        logger.error("Failed to store chunks for %s: %s", file.filename, e)
        raise HTTPException(503, "Vector database error, please try again")

    pages = len({c["page"] for c in chunks})
    return {"file_name": file.filename, "pages_with_text": pages, "chunks": len(chunks)}


@app.post("/ask")
def ask(request: AskRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(400, "Question cannot be empty")

    logger.info("Question: %s", question)

    try:
        if vector_store.count() == 0:
            raise HTTPException(400, "No documents uploaded yet")
        return rag.answer_question(question)
    except HTTPException:
        raise
    except (LLMError, rag.VectorStoreError) as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        logger.exception("Unexpected error while answering: %s", e)
        raise HTTPException(500, "Something went wrong while answering the question")

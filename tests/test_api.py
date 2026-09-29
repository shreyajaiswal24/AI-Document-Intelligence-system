import pymupdf
from fastapi.testclient import TestClient

from app import rag
from app.main import app
from app.pdf_processor import clean_text, split_text

client = TestClient(app)


def make_pdf(text):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert "vector_db" in response.json()


def test_upload_rejects_non_pdf():
    response = client.post("/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400


def test_upload_rejects_empty_file():
    response = client.post("/upload", files={"file": ("empty.pdf", b"", "application/pdf")})
    assert response.status_code == 400


def test_upload_rejects_fake_pdf():
    response = client.post("/upload", files={"file": ("fake.pdf", b"not really a pdf", "application/pdf")})
    assert response.status_code == 400


def test_upload_pdf_without_text():
    blank = make_pdf("")
    response = client.post("/upload", files={"file": ("blank.pdf", blank, "application/pdf")})
    assert response.status_code == 422


def test_ask_rejects_empty_question():
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 400


def test_no_llm_call_when_nothing_relevant(monkeypatch):
    far_away = [{"chunk_id": "x", "text": "...", "file_name": "a.pdf", "page": 1, "distance": 0.95}]
    monkeypatch.setattr(rag.vector_store, "search", lambda q: far_away)

    def fail(*args):
        raise AssertionError("LLM should not be called")

    monkeypatch.setattr(rag.llm, "generate_answer", fail)
    result = rag.answer_question("something unrelated")
    assert result["answered"] is False
    assert result["citations"] == []


def test_citations_are_parsed(monkeypatch):
    hits = [
        {"chunk_id": "a-p1-c0", "text": "Sales grew 20%.", "file_name": "a.pdf", "page": 1, "distance": 0.3},
        {"chunk_id": "a-p2-c0", "text": "Costs fell.", "file_name": "a.pdf", "page": 2, "distance": 0.4},
    ]
    monkeypatch.setattr(rag.vector_store, "search", lambda q: hits)
    monkeypatch.setattr(rag.vector_store, "add_next_chunks", lambda h: h)
    monkeypatch.setattr(rag.llm, "generate_answer", lambda q, c: "Sales grew 20% [1†L1-L2].")
    result = rag.answer_question("How much did sales grow?")
    assert result["answer"] == "Sales grew 20% [1]."
    assert [c["page"] for c in result["citations"]] == [1]


def test_split_text_overlap_and_size():
    text = "This is a sentence. " * 100
    chunks = split_text(text, size=200, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_clean_text_fixes_ligatures():
    assert clean_text("con\ufb01dent") == "confident"


def test_clean_text_removes_page_numbers():
    assert clean_text("Hello\n  12  \nworld") == "Hello\n\nworld"


def test_ask_returns_503_when_vector_db_fails(monkeypatch):
    def broken(q):
        raise RuntimeError("db down")

    monkeypatch.setattr(rag.vector_store, "count", lambda: 1)
    monkeypatch.setattr(rag.vector_store, "search", broken)
    response = client.post("/ask", json={"question": "anything"})
    assert response.status_code == 503

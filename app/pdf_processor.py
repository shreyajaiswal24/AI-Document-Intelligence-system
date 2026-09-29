import logging
import re
import unicodedata

import pymupdf

from app.config import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)


def extract_pages(pdf_bytes):
    """Return a list of (page_number, text) for every page in the PDF."""
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise ValueError(f"Could not open PDF: {e}")

    pages = []
    for i, page in enumerate(doc):
        pages.append((i + 1, page.get_text()))
    doc.close()
    return pages


def clean_text(text):
    # turns ligatures like "ﬁ" into "fi" and non-breaking spaces into normal ones
    text = unicodedata.normalize("NFKC", text)
    # join words broken with a hyphen at the end of a line
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # lines that are only a page number
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into chunks of about `size` characters.

    We try to cut at a paragraph or sentence end so chunks don't stop
    in the middle of a sentence. Neighbouring chunks share `overlap`
    characters so an answer sitting on the border is not lost.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            # best place to cut first: paragraph, sentence, line, then word
            for sep in ["\n\n", ". ", ".\n", "\n", " "]:
                cut = window.rfind(sep)
                if cut > size // 2:
                    end = start + cut + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        # step back for the overlap, but start at a word boundary
        start = end - overlap
        next_space = text.find(" ", start, end)
        if next_space != -1:
            start = next_space + 1
    return chunks


def process_pdf(pdf_bytes, file_name):
    """Extract, clean and chunk a PDF. Returns a list of chunk dicts."""
    pages = extract_pages(pdf_bytes)
    chunks = []
    for page_number, raw_text in pages:
        text = clean_text(raw_text)
        if len(text) < 30:  # empty page or just a heading
            continue
        for i, piece in enumerate(split_text(text)):
            chunks.append({
                "id": f"{file_name}-p{page_number}-c{i}",
                "text": piece,
                "file_name": file_name,
                "page": page_number,
                "chunk_index": i,
            })

    logger.info("%s: %d pages -> %d chunks", file_name, len(pages), len(chunks))
    return chunks

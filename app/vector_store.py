import logging

import chromadb
from chromadb.utils import embedding_functions

from app.config import CHROMA_DIR, COLLECTION_NAME, TOP_K

logger = logging.getLogger(__name__)

# all-MiniLM-L6-v2 (runs locally through ONNX, no API key or GPU needed)
embedder = embedding_functions.DefaultEmbeddingFunction()

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedder,
    metadata={"hnsw:space": "cosine"},
)


BATCH_SIZE = 500  # Chroma has a max batch size, so big PDFs are added in parts


def add_chunks(chunks):
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        # upsert so uploading the same file twice doesn't create duplicates
        collection.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[
                {"file_name": c["file_name"], "page": c["page"], "chunk_index": c["chunk_index"]}
                for c in batch
            ],
        )
    logger.info("Stored %d chunks (total in collection: %d)", len(chunks), collection.count())


def search(question, top_k=TOP_K):
    if collection.count() == 0:
        return []

    result = collection.query(query_texts=[question], n_results=top_k)
    hits = []
    for doc_id, text, meta, distance in zip(
        result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append({
            "chunk_id": doc_id,
            "text": text,
            "file_name": meta["file_name"],
            "page": meta["page"],
            "distance": round(distance, 3),
        })

    logger.info("Retrieved %d chunks, distances: %s", len(hits), [h["distance"] for h in hits])
    return hits


def add_next_chunks(hits):
    """Append the following chunk (same page) to each hit.

    Lists and paragraphs are often cut at a chunk border, so the start of
    the answer is in the retrieved chunk and the rest is in the next one.
    """
    next_ids = []
    for h in hits:
        prefix, index = h["chunk_id"].rsplit("-c", 1)
        next_ids.append(f"{prefix}-c{int(index) + 1}")

    found = collection.get(ids=next_ids)
    next_texts = dict(zip(found["ids"], found["documents"]))

    for h, next_id in zip(hits, next_ids):
        if next_id in next_texts:
            h["text"] = h["text"] + "\n" + next_texts[next_id]
    return hits


def count():
    return collection.count()


def list_files():
    metas = collection.get(include=["metadatas"])["metadatas"]
    return sorted({m["file_name"] for m in metas})


def delete_file(file_name):
    collection.delete(where={"file_name": file_name})
    logger.info("Deleted all chunks of %s", file_name)

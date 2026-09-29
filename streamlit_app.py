"""Simple web UI: upload PDFs and ask questions.

Run locally:  streamlit run streamlit_app.py
It uses the same pipeline as the FastAPI app (app/), just without HTTP in between.
"""
import os

import streamlit as st

# On Streamlit Cloud the key comes from the app's Secrets, locally from .env
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass  # no secrets file, .env will be used

from app import config, rag, vector_store  # noqa: E402
from app.llm import LLMError  # noqa: E402
from app.pdf_processor import process_pdf  # noqa: E402

st.set_page_config(page_title="AI Document Intelligence System", page_icon="📄")
# smaller title that shrinks with the screen, so it stays on one line
st.markdown(
    "<h1 style='white-space: nowrap; font-size: clamp(1.1rem, 4.2vw, 2.2rem);'>"
    "📄 AI Document Intelligence System</h1>",
    unsafe_allow_html=True,
)
st.caption("Upload PDFs and ask questions. Answers come only from your documents, with page citations.")

if not config.GROQ_API_KEY:
    st.error("GROQ_API_KEY is not set. Add it to .env (local) or to the app's Secrets (Streamlit Cloud).")
    st.stop()

# ---------- sidebar: upload and manage documents ----------
with st.sidebar:
    st.header("Documents")
    files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if files and st.button("Process", type="primary"):
        for file in files:
            data = file.getvalue()
            if len(data) == 0:
                st.error(f"{file.name}: the file is empty")
                continue
            if len(data) > config.MAX_FILE_MB * 1024 * 1024:
                st.error(f"{file.name}: larger than {config.MAX_FILE_MB} MB")
                continue
            try:
                with st.spinner(f"Processing {file.name}..."):
                    chunks = process_pdf(data, file.name)
                    if not chunks:
                        st.error(f"{file.name}: no text found (it may be a scanned PDF)")
                        continue
                    vector_store.add_chunks(chunks)
                st.success(f"{file.name}: {len(chunks)} chunks added")
            except ValueError as e:
                st.error(f"{file.name}: {e}")
            except Exception:
                st.error(f"{file.name}: could not be saved to the vector database")

    st.divider()
    indexed = vector_store.list_files()
    if not indexed:
        st.info("No documents yet.")
    for name in indexed:
        col1, col2 = st.columns([4, 1])
        col1.write(name)
        if col2.button("✕", key=f"del-{name}", help="Remove this document"):
            vector_store.delete_file(name)
            st.rerun()

# ---------- main: ask questions ----------
question = st.text_input("Your question", placeholder="e.g. What are the four functions of the AI RMF Core?")

if st.button("Ask") and question.strip():
    if not vector_store.list_files():
        st.warning("Please upload a document first.")
    else:
        try:
            with st.spinner("Searching the documents..."):
                result = rag.answer_question(question.strip())
        except (LLMError, rag.VectorStoreError) as e:
            st.error(str(e))
        else:
            if result["answered"]:
                st.markdown(result["answer"])
                st.subheader("Sources")
                for c in result["citations"]:
                    with st.expander(f"[{c['ref']}] {c['file_name']} — page {c['page']}"):
                        st.write(c["text"] + "...")
            else:
                st.warning(result["answer"])

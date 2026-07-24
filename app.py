"""Streamlit UI for the Semantic Search Engine.

Sentence Transformers + FAISS. Upload a plain-text file of alternating
question/answer lines and search it semantically.
"""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import streamlit as st

from src.config import settings
from src.document_loader import DocumentLoadError, load_qa_pairs
from src.embeddings import EmbeddingModel
from src.logger import get_logger
from src.search_engine import SemanticSearchEngine

logger = get_logger(__name__)

st.set_page_config(page_title="Semantic Search Engine", page_icon="🔍", layout="wide")

# -----------------------------
# STYLES
# -----------------------------
st.markdown(
    """
    <style>
    .main { background-color: #0f172a; color: white; }
    .stTextInput > div > div > input {
        background-color: #1e293b; color: white; border-radius: 10px;
    }
    .stFileUploader { background-color: #1e293b; padding: 15px; border-radius: 12px; }
    .result-box {
        background-color: #1e293b; padding: 18px; border-radius: 12px;
        margin-bottom: 15px; border: 1px solid #334155;
    }
    .score { color: #38bdf8; font-size: 18px; font-weight: bold; }
    .rank-badge {
        display: inline-block; background-color: #38bdf8; color: #0f172a;
        border-radius: 999px; padding: 2px 10px; font-weight: bold; margin-right: 8px;
    }
    .title-style { text-align: center; color: #38bdf8; font-size: 42px; font-weight: bold; }
    .subtitle { text-align: center; color: #cbd5e1; margin-bottom: 30px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel()


@st.cache_resource(show_spinner=False)
def build_search_engine(file_hash: str, filepath: str) -> SemanticSearchEngine:
    """Build (and cache) a search engine for a given file.

    Cached by content hash rather than filename, so re-running the app
    or uploading the same file twice doesn't waste time rebuilding an
    identical index, while a genuinely new/edited file still triggers
    a rebuild.
    """
    model = get_embedding_model()
    engine = SemanticSearchEngine(model)
    documents = load_qa_pairs(filepath)
    engine.build_index(documents)
    return engine


def render_results(results, query: str) -> None:
    st.markdown(f"## 🔎 Results for _{query}_")
    if not results:
        st.warning("No matching documents found.")
        return

    for result in results:
        st.markdown(
            f"""
            <div class="result-box">
                <span class="rank-badge">#{result['rank']}</span>
                <span class="score">Distance: {result['score']:.4f}</span>
                <br><br>
                <div style="white-space: pre-wrap;">{result['document']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    st.markdown('<div class="title-style">🔍 Semantic Search Engine</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Sentence Transformers + FAISS + Streamlit</div>',
        unsafe_allow_html=True,
    )

    st.sidebar.title("⚙️ Settings")
    top_k = st.sidebar.slider(
        "Number of results",
        min_value=1,
        max_value=settings.max_top_k,
        value=settings.default_top_k,
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Embedding model: `{settings.embedding_model_name}`")
    st.sidebar.caption("Index type: FAISS FlatL2 (exact search)")

    uploaded_file = st.file_uploader(
        "📄 Upload a .txt file containing alternating Q&A lines",
        type=["txt"],
    )

    if uploaded_file is None:
        st.info("📌 Please upload a .txt file to begin.")
        st.caption(
            "Expected format: one question per line, immediately followed by its answer line."
        )
        return

    file_bytes = uploaded_file.getbuffer()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        st.error(f"File too large ({size_mb:.1f} MB). Limit is {settings.max_upload_size_mb} MB.")
        return

    file_hash = hashlib.sha256(file_bytes).hexdigest()

    with tempfile.TemporaryDirectory() as tmp_dir:
        filepath = Path(tmp_dir) / uploaded_file.name
        with open(filepath, "wb") as f:
            f.write(file_bytes)

        try:
            with st.spinner("Preparing model and building index..."):
                engine = build_search_engine(file_hash, str(filepath))
        except DocumentLoadError as exc:
            st.error(f"❌ {exc}")
            return
        except Exception:
            logger.exception("Unexpected error while building the index")
            st.error("❌ An unexpected error occurred while processing your file.")
            return

    st.success(f"✅ Indexed {engine.document_count} documents")

    query = st.text_input(
        "💬 Enter your search query",
        placeholder="Example: What is machine learning?",
    )

    if query:
        try:
            with st.spinner("Searching..."):
                results = engine.search(query, top_k=top_k)
        except Exception:
            logger.exception("Search failed")
            st.error("❌ Something went wrong while searching. Please try again.")
            return

        render_results(results, query)


if __name__ == "__main__":
    main()



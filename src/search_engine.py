"""Semantic search engine backed by a FAISS similarity index."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, TypedDict, Union

import faiss
import numpy as np

from src.embeddings import EmbeddingModel
from src.logger import get_logger

logger = get_logger(__name__)

# Unit separator: safe delimiter for joining/splitting documents on disk,
# since real Q&A text is very unlikely to contain this control character.
_DOC_DELIMITER = "\u241E"


class SearchResult(TypedDict):
    document: str
    score: float
    rank: int


class SemanticSearchEngine:
    """Builds and queries a FAISS L2 index over a set of documents."""

    def __init__(self, embedding_model: EmbeddingModel) -> None:
        self._embedding_model = embedding_model
        self._index: Optional[faiss.IndexFlatL2] = None
        self._documents: List[str] = []

    @property
    def is_ready(self) -> bool:
        return self._index is not None and self._index.ntotal > 0

    @property
    def document_count(self) -> int:
        return len(self._documents)

    def build_index(self, documents: List[str]) -> None:
        if not documents:
            raise ValueError("Cannot build an index from an empty document list.")

        logger.info("Generating embeddings for %d documents", len(documents))
        embeddings = self._embedding_model.encode(documents)

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        self._index = index
        self._documents = documents
        logger.info("FAISS index built with %d vectors", index.ntotal)

    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        if not self.is_ready:
            raise RuntimeError("Index has not been built yet. Call build_index() first.")
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        top_k = min(top_k, self.document_count)
        query_embedding = self._embedding_model.encode([query])

        distances, indices = self._index.search(query_embedding, top_k)

        results: List[SearchResult] = []
        for rank, (doc_idx, distance) in enumerate(zip(indices[0], distances[0]), start=1):
            if doc_idx == -1:
                continue
            results.append(
                SearchResult(
                    document=self._documents[doc_idx],
                    score=float(distance),
                    rank=rank,
                )
            )
        return results

    def save(self, directory: Union[str, Path]) -> None:
        """Persist the FAISS index and documents to disk for reuse."""
        if not self.is_ready:
            raise RuntimeError("Cannot save an index that has not been built.")

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / "index.faiss"))
        (directory / "documents.txt").write_text(
            _DOC_DELIMITER.join(self._documents), encoding="utf-8"
        )
        logger.info("Index and documents saved to %s", directory)

    def load(self, directory: Union[str, Path]) -> None:
        """Load a previously saved FAISS index and its documents."""
        directory = Path(directory)
        self._index = faiss.read_index(str(directory / "index.faiss"))
        self._documents = (
            (directory / "documents.txt").read_text(encoding="utf-8").split(_DOC_DELIMITER)
        )
        logger.info("Index and documents loaded from %s", directory)

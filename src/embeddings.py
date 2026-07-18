"""Embedding generation wrapper around SentenceTransformers."""
from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)


class EmbeddingModel:
    """Loads a SentenceTransformer model once and exposes a consistent
    encode() interface returning float32 numpy arrays, which is the
    dtype FAISS expects.
    """

    def __init__(self, model_name: str = settings.embedding_model_name) -> None:
        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dim = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            raise ValueError("Cannot encode an empty list of texts.")
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return np.asarray(embeddings, dtype="float32")

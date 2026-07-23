"""Configuration management for the Semantic Search Engine.

Centralizes all tunable parameters so they aren't scattered as magic
numbers/strings throughout the codebase. Values can be overridden via
environment variables, which keeps the app container/cloud friendly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Application-wide settings."""

    # Embedding model
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

    # Search defaults
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "3"))
    max_top_k: int = int(os.getenv("MAX_TOP_K", "10"))

    # File handling
    max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))

    # FAISS
    index_type: str = os.getenv("FAISS_INDEX_TYPE", "flat_l2")

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()



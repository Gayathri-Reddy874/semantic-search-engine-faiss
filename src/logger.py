"""Centralized logging configuration."""
from __future__ import annotations

import logging
import sys

from src.config import settings


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger, safe to call repeatedly without
    duplicating handlers (which would otherwise double-print logs)."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(settings.log_level)
        logger.propagate = False

    return logger

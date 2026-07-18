"""Document loading and parsing utilities.

Currently supports plain-text files containing alternating
question/answer lines. Kept as a standalone module so additional
parsers (CSV, JSON, PDF, ...) can be added later without touching
calling code in app.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Union


class DocumentLoadError(Exception):
    """Raised when a document file cannot be parsed into valid documents."""


def load_qa_pairs(filepath: Union[str, Path]) -> List[str]:
    """Load a text file of alternating question/answer lines.

    Each pair of non-empty lines is combined into a single document
    formatted as "Q: ...\\nA: ...". Files with an odd number of lines
    have their final question paired with an empty answer.

    Args:
        filepath: Path to a UTF-8 encoded .txt file.

    Returns:
        A list of formatted Q&A document strings.

    Raises:
        DocumentLoadError: If the file is missing, empty, or unreadable.
    """
    path = Path(filepath)

    if not path.exists():
        raise DocumentLoadError(f"File not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as file:
            lines = [line.strip() for line in file if line.strip()]
    except UnicodeDecodeError as exc:
        raise DocumentLoadError(
            "File is not valid UTF-8 text. Please upload a plain .txt file."
        ) from exc

    if not lines:
        raise DocumentLoadError("File is empty or contains no readable text.")

    documents: List[str] = []
    for i in range(0, len(lines), 2):
        question = lines[i]
        answer = lines[i + 1] if i + 1 < len(lines) else ""
        documents.append(f"Q: {question}\nA: {answer}")

    return documents

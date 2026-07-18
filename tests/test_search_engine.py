"""Unit tests for src.search_engine.

Uses a lightweight fake embedding model so tests run fast and offline,
without downloading a real SentenceTransformer checkpoint.
"""
import numpy as np
import pytest

from src.search_engine import SemanticSearchEngine


class FakeEmbeddingModel:
    """Deterministic fake: hashes each string to seed a fixed-length
    vector, so tests exercise the FAISS/search plumbing without needing
    a real model or network access."""

    def encode(self, texts):
        vectors = []
        for text in texts:
            rng = np.random.RandomState(abs(hash(text)) % (2**32))
            vectors.append(rng.rand(8).astype("float32"))
        return np.vstack(vectors)


@pytest.fixture
def engine():
    return SemanticSearchEngine(FakeEmbeddingModel())


def test_build_index_and_is_ready(engine):
    engine.build_index(["doc one", "doc two", "doc three"])
    assert engine.is_ready
    assert engine.document_count == 3


def test_build_index_empty_raises(engine):
    with pytest.raises(ValueError):
        engine.build_index([])


def test_search_before_build_raises(engine):
    with pytest.raises(RuntimeError):
        engine.search("hello")


def test_search_returns_ranked_results(engine):
    engine.build_index(["apple pie recipe", "car engine repair", "banana smoothie"])
    results = engine.search("apple pie recipe", top_k=2)

    assert len(results) == 2
    assert results[0]["rank"] == 1
    assert results[0]["document"] == "apple pie recipe"


def test_search_empty_query_raises(engine):
    engine.build_index(["doc one"])
    with pytest.raises(ValueError):
        engine.search("   ")


def test_top_k_capped_at_document_count(engine):
    engine.build_index(["doc one", "doc two"])
    results = engine.search("doc one", top_k=10)
    assert len(results) == 2


def test_save_and_load_roundtrip(engine, tmp_path):
    engine.build_index(["doc one", "doc two"])
    engine.save(tmp_path / "saved_index")

    reloaded = SemanticSearchEngine(FakeEmbeddingModel())
    reloaded.load(tmp_path / "saved_index")

    assert reloaded.is_ready
    assert reloaded.document_count == 2

"""Unit tests for src.document_loader."""
import pytest

from src.document_loader import DocumentLoadError, load_qa_pairs


def test_load_qa_pairs_even_lines(tmp_path):
    file = tmp_path / "qa.txt"
    file.write_text("What is AI?\nArtificial Intelligence.\nWhat is ML?\nMachine Learning.")

    documents = load_qa_pairs(file)

    assert len(documents) == 2
    assert documents[0] == "Q: What is AI?\nA: Artificial Intelligence."
    assert documents[1] == "Q: What is ML?\nA: Machine Learning."


def test_load_qa_pairs_odd_lines_pairs_last_question_with_empty_answer(tmp_path):
    file = tmp_path / "qa.txt"
    file.write_text("What is AI?\nArtificial Intelligence.\nWhat is NLP?")

    documents = load_qa_pairs(file)

    assert len(documents) == 2
    assert documents[1] == "Q: What is NLP?\nA: "


def test_load_qa_pairs_skips_blank_lines(tmp_path):
    file = tmp_path / "qa.txt"
    file.write_text("Q1\n\nA1\n\nQ2\nA2\n")

    documents = load_qa_pairs(file)

    assert len(documents) == 2


def test_missing_file_raises(tmp_path):
    with pytest.raises(DocumentLoadError):
        load_qa_pairs(tmp_path / "does_not_exist.txt")


def test_empty_file_raises(tmp_path):
    file = tmp_path / "empty.txt"
    file.write_text("")

    with pytest.raises(DocumentLoadError):
        load_qa_pairs(file)


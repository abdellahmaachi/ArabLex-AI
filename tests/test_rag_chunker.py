"""Segmentation, provenance et intégration avec l'ingestion RAG."""

from copy import deepcopy
from uuid import UUID

import pytest

from src.rag.chunker import chunk_document, chunk_documents
from src.rag.ingestion import prepare_documents


def test_overlapping_windows_preserve_arabic_and_offsets():
    text = "  إِنَّ المحكمة\n\nلا تقبل\t١٢/٣. أبدا  "
    chunks = chunk_document({"id": "law", "text": text}, chunk_size=3, overlap=1)
    assert [chunk["text"] for chunk in chunks] == [
        "إِنَّ المحكمة\n\nلا", "لا تقبل\t١٢/٣.", "١٢/٣. أبدا",
    ]
    assert [chunk["word_count"] for chunk in chunks] == [3, 3, 2]
    assert [chunk["chunk_index"] for chunk in chunks] == [0, 1, 2]
    for chunk in chunks:
        assert text[chunk["start"]:chunk["end"]] == chunk["text"]
        assert chunk["text_field"] == "text"


@pytest.mark.parametrize("count", range(1, 16))
@pytest.mark.parametrize("overlap", [0, 1, 3])
def test_complete_coverage_and_no_redundant_tail(count, overlap):
    words = [str(index) for index in range(count)]
    chunks = chunk_document({"id": "a", "text": " ".join(words)}, chunk_size=4, overlap=overlap)
    recovered = chunks[0]["text"].split()
    for previous, current in zip(chunks, chunks[1:]):
        if overlap:
            assert previous["text"].split()[-overlap:] == current["text"].split()[:overlap]
        recovered.extend(current["text"].split()[overlap:])
    assert recovered == words
    assert all(chunk["word_count"] <= 4 for chunk in chunks)
    assert len(chunks) == 1 or chunks[-1]["word_count"] > overlap


def test_metadata_copies_and_prepared_text_integration():
    source = {
        "id": "law", "text": "\ufeff  أ ب\r\nج د  ", "class": "real_estate",
        "filename": "law.txt", "path": "/corpus/law.txt", "extra": {"tags": ["law"]},
    }
    document = prepare_documents([source])[0]
    before = deepcopy(document)
    chunks = chunk_documents([document], chunk_size=2, overlap=0)
    assert [chunk["text"] for chunk in chunks] == ["أ ب", "ج د"]
    for chunk in chunks:
        assert chunk["document_id"] == "law"
        assert chunk["text_field"] == "prepared_text"
        assert document["prepared_text"][chunk["start"]:chunk["end"]] == chunk["text"]
        assert chunk["metadata"] == {key: source[key] for key in ("class", "filename", "path", "extra")}
    chunks[0]["metadata"]["extra"]["tags"].append("changed")
    assert chunks[1]["metadata"]["extra"]["tags"] == ["law"]
    assert document == before


def test_ids_are_stable_unique_and_sensitive_to_content():
    documents = [{"id": "z", "text": "a b a b"}, {"id": "a", "text": "a b"}]
    chunks = chunk_documents(iter(documents), chunk_size=2, overlap=0)
    assert chunks == chunk_documents(documents, chunk_size=2, overlap=0)
    assert [chunk["document_id"] for chunk in chunks] == ["z", "z", "a"]
    assert len({chunk["id"] for chunk in chunks}) == 3
    assert all(UUID(chunk["id"]).version == 5 for chunk in chunks)
    changed = chunk_document({"id": "z", "text": "c d a b"}, chunk_size=2, overlap=0)
    assert changed[0]["id"] != chunks[0]["id"]
    assert changed[1]["id"] == chunks[1]["id"]


@pytest.mark.parametrize("text", ["", " \t\r\n"])
def test_empty_text(text):
    assert chunk_document({"id": "a", "text": text}) == []
    assert chunk_document({"id": "a", "text": "original", "prepared_text": text}) == []
    assert chunk_documents([]) == []


@pytest.mark.parametrize("size,overlap,error", [
    (0, 0, ValueError), (-1, 0, ValueError), (2, -1, ValueError),
    (2, 2, ValueError), (2, 3, ValueError), (True, 0, TypeError),
    (2.5, 0, TypeError), (2, False, TypeError), (2, "1", TypeError),
])
def test_invalid_options_even_for_empty_corpus(size, overlap, error):
    with pytest.raises(error):
        chunk_documents([], chunk_size=size, overlap=overlap)


@pytest.mark.parametrize("document,error", [
    (None, TypeError), ({}, ValueError), ({"id": " "}, ValueError),
    ({"id": 1}, ValueError), ({"id": "a"}, TypeError),
    ({"id": "a", "text": None}, TypeError),
    ({"id": "a", "text": "valid", "prepared_text": None}, TypeError),
])
def test_invalid_documents(document, error):
    with pytest.raises(error):
        chunk_documents([document])


def test_duplicate_document_ids():
    with pytest.raises(ValueError, match="dupliqué"):
        chunk_documents([{"id": "a", "text": ""}, {"id": "a", "text": "valid"}])

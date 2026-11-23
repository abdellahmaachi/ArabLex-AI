"""Récupération RAG sans téléchargement ni inférence de modèle réel."""

from unittest.mock import Mock

import numpy as np
import pytest

from src.rag.chunker import chunk_documents
from src.rag.qdrant_store import QdrantStore
from src.rag.retriever import RAGRetriever


def components():
    encoder = Mock(dimension=2)
    encoder.encode_query.return_value = np.array([1, 0], dtype=np.float32)
    store = Mock(dimension=2)
    store.search.return_value = []
    return encoder, store


def test_preserves_query_and_forwards_search_options():
    encoder, store = components()
    expected = [{"id": "id", "score": 0.9, "passage": {"text": "نص", "metadata": {}}}]
    store.search.return_value = expected
    query = "  ما هي حقوق الملكية؟\n"
    result = RAGRetriever(encoder, store).retrieve(
        query, top_k=3, document_id="law", class_name="real_estate", score_threshold=0.4,
    )
    encoder.encode_query.assert_called_once_with(query)
    store.search.assert_called_once_with(
        encoder.encode_query.return_value, top_k=3,
        document_id="law", class_name="real_estate", score_threshold=0.4,
    )
    assert result == expected


@pytest.mark.parametrize("query,kwargs,error", [
    (None, {}, TypeError), ("", {}, ValueError), (" \n", {}, ValueError),
    ("نص", {"top_k": 0}, ValueError), ("نص", {"top_k": True}, ValueError),
    ("نص", {"top_k": 1.5}, ValueError), ("نص", {"document_id": ""}, ValueError),
    ("نص", {"class_name": 1}, ValueError), ("نص", {"score_threshold": True}, TypeError),
    ("نص", {"score_threshold": "0.5"}, TypeError),
    ("نص", {"score_threshold": float("nan")}, ValueError),
    ("نص", {"score_threshold": float("inf")}, ValueError),
    ("نص", {"score_threshold": -2}, ValueError),
])
def test_invalid_input_never_runs_inference(query, kwargs, error):
    encoder, store = components()
    with pytest.raises(error):
        RAGRetriever(encoder, store).retrieve(query, **kwargs)
    encoder.encode_query.assert_not_called()
    store.search.assert_not_called()


@pytest.mark.parametrize("dimension", [0, True, 1.5, 3])
def test_rejects_incompatible_dimensions(dimension):
    encoder, store = components()
    store.dimension = dimension
    with pytest.raises(ValueError):
        RAGRetriever(encoder, store)


def test_errors_propagate():
    encoder, store = components()
    retriever = RAGRetriever(encoder, store)
    encoder.encode_query.side_effect = RuntimeError("model failed")
    with pytest.raises(RuntimeError, match="model failed"):
        retriever.retrieve("نص")
    store.search.assert_not_called()
    encoder.encode_query.side_effect = None
    store.search.side_effect = RuntimeError("store closed")
    with pytest.raises(RuntimeError, match="store closed"):
        retriever.retrieve("نص")


def test_top_k_filters_scores_and_sources_with_qdrant():
    encoder, _ = components()
    passages = chunk_documents([
        {"id": "law", "text": "أ ب ج د", "class": "real_estate", "filename": "law.txt"},
        {"id": "other", "text": "هـ و", "class": "family_inheritance"},
    ], chunk_size=2, overlap=0)
    with QdrantStore(dimension=2) as store:
        retriever = RAGRetriever(encoder, store)
        assert retriever.retrieve("نص") == []
        store.upsert(passages, [[1, 0], [0.8, 0.6], [0, 1]])
        hits = retriever.retrieve("نص", top_k=2)
        assert [hit["id"] for hit in hits] == [p["id"] for p in passages[:2]]
        np.testing.assert_allclose([hit["score"] for hit in hits], [1, 0.8], atol=1e-6)
        assert [hit["passage"] for hit in hits] == passages[:2]
        assert len(retriever.retrieve("نص", top_k=10)) == 3
        assert len(retriever.retrieve("نص", score_threshold=0.9)) == 1
        assert len(retriever.retrieve("نص", document_id="other")) == 1
        assert len(retriever.retrieve("نص", class_name="real_estate")) == 2
        assert retriever.retrieve("نص", document_id="other", class_name="real_estate") == []

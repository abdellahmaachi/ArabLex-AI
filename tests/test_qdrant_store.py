"""Tests d'intégration Qdrant Local sans serveur et sans modèle NLP."""

from copy import deepcopy

import numpy as np
import pytest

from src.rag.chunker import chunk_documents
from src.rag.qdrant_store import QdrantStore


@pytest.fixture
def passages():
    return chunk_documents([
        {"id": "a", "text": "نص قانوني", "class": "real_estate"},
        {"id": "b", "text": "حكم قضائي", "class": "family_inheritance"},
        {"id": "c", "text": "حقوق الملكية", "class": "real_estate"},
    ])


def test_persistence_ranking_and_payload(tmp_path, passages):
    before = deepcopy(passages)
    with QdrantStore(tmp_path / "index", dimension=2) as store:
        assert store.search([1, 0]) == []
        assert store.upsert(iter(passages), [[1, 0], [0, 1], [-1, 0]], batch_size=2) == 3
        assert store.count() == 3
        hits = store.search([3, 0], top_k=10)
        assert [hit["passage"]["document_id"] for hit in hits] == ["a", "b", "c"]
        np.testing.assert_allclose([hit["score"] for hit in hits], [1, 0, -1], atol=1e-6)
        assert hits[0]["passage"] == passages[0]
        hits[0]["passage"]["metadata"]["class"] = "modified"
        assert store.search([1, 0])[0]["passage"] == passages[0]
    with QdrantStore(tmp_path / "index", dimension=2) as reopened:
        assert reopened.count() == 3
        assert reopened.search([1, 0], top_k=1)[0]["id"] == passages[0]["id"]
    assert passages == before


def test_upsert_filters_threshold_and_deletion(passages):
    with QdrantStore(dimension=2) as store:
        vectors = [[1, 0], [0, 1], [-1, 0]]
        store.upsert(passages, vectors)
        store.upsert(passages, vectors)
        assert store.count() == 3
        assert len(store.search([1, 0], class_name="real_estate")) == 2
        assert store.search([1, 0], class_name="real_estate", document_id="b") == []
        assert len(store.search([1, 0], document_id="b")) == 1
        assert len(store.search([1, 0], score_threshold=0.5)) == 1
        updated = deepcopy(passages[0])
        updated["text"] = "نص معدل"
        store.upsert([updated], [[0, 1]])
        assert store.search([0, 1], document_id="a")[0]["passage"]["text"] == "نص معدل"
        assert store.count() == 3
        store.delete_document("a")
        assert store.count() == 2
        assert store.search([1, 0], document_id="a") == []
        store.delete_document("missing")
        assert store.count() == 2


def test_incompatible_collection_is_preserved_and_unlocked(tmp_path, passages):
    with QdrantStore(tmp_path, dimension=2) as store:
        store.upsert(passages[:1], [[1, 0]])
    with pytest.raises(ValueError, match="incompatible"):
        QdrantStore(tmp_path, dimension=3)
    with QdrantStore(tmp_path, dimension=2) as store:
        assert store.count() == 1


@pytest.mark.parametrize("vectors", [
    [[1, 0]], [[1, 0, 0]] * 3, [[0, 0]] * 3,
    [[float("nan"), 1]] * 3, [[float("inf"), 1]] * 3,
])
def test_invalid_embeddings_do_not_write(passages, vectors):
    with QdrantStore(dimension=2) as store:
        with pytest.raises(ValueError):
            store.upsert(passages, vectors)
        assert store.count() == 0


@pytest.mark.parametrize("mutation", ["duplicate", "id", "payload", "text"])
def test_whole_input_validated_before_first_batch(passages, mutation):
    if mutation == "duplicate":
        passages[-1]["id"] = passages[0]["id"]
    elif mutation == "id":
        passages[-1]["id"] = "not-a-uuid"
    elif mutation == "payload":
        passages[-1]["metadata"]["bad"] = float("nan")
    else:
        passages[-1]["text"] = ""
    with QdrantStore(dimension=2) as store:
        with pytest.raises(ValueError):
            store.upsert(passages, [[1, 0]] * 3, batch_size=1)
        assert store.count() == 0


def test_empty_input_closed_store_and_search_validation():
    store = QdrantStore(dimension=2)
    with store:
        assert store.upsert([], np.empty((0, 2))) == 0
        for vector in ([0, 0], [1], [float("nan"), 1]):
            with pytest.raises(ValueError):
                store.search(vector)
        for kwargs in ({"top_k": 0}, {"top_k": True}, {"score_threshold": 2},
                       {"score_threshold": float("nan")}, {"document_id": ""}):
            with pytest.raises(ValueError):
                store.search([1, 0], **kwargs)
    store.close()
    with pytest.raises(RuntimeError, match="fermé"):
        store.count()


def test_invalid_constructor():
    for dimension in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            QdrantStore(dimension=dimension)

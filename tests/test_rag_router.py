"""Contrat HTTP RAG et cycle de vie, sans réseau ni checkpoint NLP."""

import json
from contextlib import ExitStack
from dataclasses import asdict
from unittest.mock import MagicMock, Mock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.app.core.config import Settings, get_settings
from backend.app.main import app
from backend.app.routers import rag
from src.rag.gemini_client import GeminiRequestError, GeminiResponse, GeminiResponseError
from src.rag.ingestion import DEFAULT_NORMALIZATION
from src.rag.qdrant_store import QdrantStore
from src.rag.service import InvalidCitationError, RAGService


@pytest.fixture
def client():
    settings = Settings(_env_file=None, rag_top_k=3, gemini_api_key=None, gemini_model=None)
    app.dependency_overrides[get_settings] = lambda: settings
    rag.close_rag_service()
    try:
        with TestClient(app) as http:
            yield http
    finally:
        app.dependency_overrides.clear()
        rag.close_rag_service()


def test_answer_without_passages_and_forwarded_options(client):
    retriever, generator = Mock(), Mock()
    retriever.retrieve.return_value = []
    service = RAGService(retriever, generator)
    app.dependency_overrides[rag.get_rag_service] = lambda: service
    response = client.post("/api/rag/ask", json={"question": "سؤال"})
    assert response.status_code == 200
    assert response.json()["citation_status"] == "not_applicable"
    assert response.json()["sources"] == []
    assert response.json()["generation"] is None
    generator.generate.assert_not_called()
    retriever.retrieve.assert_called_with(
        "سؤال", top_k=3, document_id=None, class_name=None, score_threshold=None,
    )
    assert client.post("/api/rag/ask", json={
        "question": "سؤال", "top_k": 2, "document_id": "d1",
        "class_name": "civil", "score_threshold": 0.5,
    }).status_code == 200
    retriever.retrieve.assert_called_with(
        "سؤال", top_k=2, document_id="d1", class_name="civil", score_threshold=0.5,
    )


def test_answer_with_sources(client):
    retriever, generator = Mock(), Mock()
    retriever.retrieve.return_value = [{
        "id": "p1", "score": 0.9, "passage": {
            "id": "p1", "document_id": "d1", "chunk_index": 0,
            "text": "نص", "text_field": "prepared_text", "start": 0, "end": 2,
            "word_count": 1, "metadata": {"class": "civil", "filename": "d1.txt"},
        },
    }]
    generator.generate.return_value = GeminiResponse(
        text="جواب [S1]", model="test", finish_reason="STOP",
    )
    service = RAGService(retriever, generator)
    app.dependency_overrides[rag.get_rag_service] = lambda: service
    response = client.post("/api/rag/ask", json={"question": "سؤال"})
    assert response.status_code == 200
    assert response.json()["sources"][0]["cited"] is True
    assert response.json()["citation_status"] == "valid"
    assert response.json()["generation"]["model"] == "test"


@pytest.mark.parametrize("payload", [
    {}, {"question": " "}, {"question": "سؤال", "top_k": 0},
    {"question": "سؤال", "score_threshold": 2},
])
def test_invalid_requests(client, payload):
    service = Mock()
    app.dependency_overrides[rag.get_rag_service] = lambda: service
    assert client.post("/api/rag/ask", json=payload).status_code == 422
    service.answer.assert_not_called()


@pytest.mark.parametrize("error,status", [
    (GeminiRequestError("secret", status_code=429, retryable=True), 503),
    (GeminiRequestError("secret", status_code=403, retryable=False), 502),
    (GeminiResponseError("secret"), 502),
    (InvalidCitationError("secret"), 502),
    (ValueError("secret"), 422),
])
def test_errors_do_not_expose_details(client, error, status):
    service = Mock()
    service.answer.side_effect = error
    app.dependency_overrides[rag.get_rag_service] = lambda: service
    response = client.post("/api/rag/ask", json={"question": "سؤال"})
    assert response.status_code == status
    assert "secret" not in response.text


def test_missing_configuration(client):
    response = client.post("/api/rag/ask", json={"question": "سؤال"})
    assert response.status_code == 503
    assert response.json() == {"detail": "Service RAG indisponible"}


def test_lifecycle_caches_and_closes_resources(monkeypatch):
    cleanup = Mock()
    service = Mock()

    def build(settings, resources):
        resources.callback(cleanup)
        return service

    monkeypatch.setattr(rag, "_build_service", build)
    rag.close_rag_service()
    settings = Settings(_env_file=None)
    with TestClient(app):
        assert rag.get_rag_service(settings) is service
        assert rag.get_rag_service(settings) is service
        cleanup.assert_not_called()
    cleanup.assert_called_once_with()


def test_failed_initialization_closes_resources_and_can_retry(monkeypatch):
    cleanup = Mock()

    def fail(settings, resources):
        resources.callback(cleanup)
        raise RuntimeError("secret")

    rag.close_rag_service()
    monkeypatch.setattr(rag, "_build_service", fail)
    for _ in range(2):
        with pytest.raises(HTTPException) as error:
            rag.get_rag_service(Settings(_env_file=None))
        assert error.value.status_code == 503
    assert cleanup.call_count == 2


def test_build_uses_published_model_and_closes_clients(tmp_path, monkeypatch):
    build = tmp_path / "build-one"
    build.mkdir()
    store_path = build / "qdrant"
    manifest_path = build / "manifest.json"
    model_path = tmp_path / "published-model"
    manifest = {
        "schema_version": 1, "distance": "Cosine", "dimension": 4,
        "collection_name": "published", "qdrant_path": str(store_path),
        "model_dir": str(model_path), "max_length": 128, "truncate": False,
        "normalization": asdict(DEFAULT_NORMALIZATION),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    pointer = tmp_path / "current.json"
    pointer.write_text(json.dumps({
        "schema_version": 1, "build_dir": str(build), "manifest": str(manifest_path),
        "qdrant_path": str(store_path), "collection_name": "published", "dimension": 4,
    }), encoding="utf-8")
    encoder = Mock(dimension=4)
    encoder_factory = Mock(return_value=encoder)
    store, generator = MagicMock(), MagicMock()
    store.__enter__.return_value = store
    store.dimension = 4
    generator.__enter__.return_value = generator
    store_factory = Mock(return_value=store)
    generator_factory = Mock(return_value=generator)
    monkeypatch.setattr(rag, "AraBERTEncoder", encoder_factory)
    monkeypatch.setattr(rag, "QdrantStore", store_factory)
    monkeypatch.setattr(rag, "GeminiClient", generator_factory)
    settings = Settings(
        _env_file=None, rag_index_path=pointer, gemini_api_key="test", gemini_model="test-model",
    )
    with ExitStack() as resources:
        service = rag._build_service(settings, resources)
        assert isinstance(service, RAGService)
        service.retriever.encoder.encode_query("\ufeff سؤال\r\n")
        encoder.encode_query.assert_called_once_with("سؤال")
        encoder_factory.assert_called_once_with(
            str(model_path), device=settings.inference_device,
            batch_size=settings.rag_batch_size, max_length=128, truncate=False,
        )
        store_factory.assert_called_once_with(
            store_path, dimension=4, collection_name="published", create_if_missing=False,
        )
    store.__exit__.assert_called_once()
    generator.__exit__.assert_called_once()


def test_existing_qdrant_required(tmp_path):
    path = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        QdrantStore(path, dimension=4, create_if_missing=False)
    assert not path.exists()
    with QdrantStore(path, dimension=4, collection_name="existing"):
        pass
    with pytest.raises(ValueError, match="introuvable"):
        QdrantStore(path, dimension=4, collection_name="missing", create_if_missing=False)
    with QdrantStore(path, dimension=4, collection_name="existing", create_if_missing=False) as store:
        assert store.count() == 0

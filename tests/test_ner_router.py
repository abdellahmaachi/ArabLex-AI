"""Recherche HTTP dans un véritable export NER, sans chargement d'AraBERT."""

import json
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings, get_settings
from backend.app.main import app
from backend.app.routers.ner import _load_service, get_ner_service
from src.ner.corpus_indexer import export_annotations
from src.ner.service import NERService


@pytest.fixture
def documents():
    return [
        {"id": "a", "class": "civil", "filename": "a.txt", "path": "a.txt",
         "text": "محمد محمد", "entities": [
             {"text": "محمد", "label": "PER", "start": 0, "end": 4},
             {"text": "محمد", "label": "PER", "start": 5, "end": 9},
         ]},
        {"id": "b", "class": "family", "filename": "b.txt", "path": "b.txt",
         "text": "محمد", "entities": [
             {"text": "محمد", "label": "ORG", "start": 0, "end": 4},
         ]},
    ]


@pytest.fixture
def client(tmp_path):
    settings = Settings(_env_file=None, ner_index_path=tmp_path / "index.json")
    app.dependency_overrides[get_settings] = lambda: settings
    _load_service.cache_clear()
    try:
        with TestClient(app) as http:
            yield http, settings
    finally:
        app.dependency_overrides.clear()
        _load_service.cache_clear()


def test_search_export_and_filters(client, documents):
    http, settings = client
    export_annotations(documents, settings.ner_index_path)
    response = http.post("/api/ner/search", json={"text": "محمد"})
    assert response.status_code == 200
    hits = response.json()["results"]
    assert [hit["id"] for hit in hits] == ["a", "b"]
    assert hits[0]["class"] == "civil"
    assert hits[0]["entities"] == documents[0]["entities"]
    for options, ids in [
        ({"label": "ORG"}, ["b"]), ({"class_name": "civil"}, ["a"]),
        ({"k": 1}, ["a"]), ({"label": "PER", "class_name": "family"}, []),
        ({"label": "UNKNOWN"}, []),
    ]:
        response = http.post("/api/ner/search", json={"text": "محمد", **options})
        assert response.status_code == 200
        assert [hit["id"] for hit in response.json()["results"]] == ids
    assert http.post("/api/ner/search", json={"text": "مُحمد"}).json()["results"] == []


def test_cache_reload_after_export(client, documents):
    _, settings = client
    export_annotations(documents, settings.ner_index_path)
    first = get_ner_service(settings)
    assert get_ner_service(settings) is first
    export_annotations([], settings.ner_index_path)
    updated = get_ner_service(settings)
    assert updated is not first
    assert updated.search("محمد")["results"] == []


def test_search_results_do_not_mutate_index(documents):
    service = NERService(documents)
    service.search("محمد")["results"][0]["entities"][0]["start"] = 99
    assert service.search("محمد")["results"][0]["entities"][0]["start"] == 0


@pytest.mark.parametrize("payload", [
    {}, {"text": " "}, {"text": "محمد", "k": 0}, {"text": "محمد", "k": True},
    {"text": "محمد", "label": " "}, {"text": "محمد", "unknown": True},
])
def test_invalid_request_does_not_search(client, payload):
    http, _ = client
    service = Mock()
    app.dependency_overrides[get_ner_service] = lambda: service
    assert http.post("/api/ner/search", json=payload).status_code == 422
    service.search.assert_not_called()


@pytest.mark.parametrize("state", ["missing", "corrupt", "inconsistent"])
def test_unavailable_index(client, documents, state):
    http, settings = client
    if state == "corrupt":
        settings.ner_index_path.write_text("invalid JSON", encoding="utf-8")
    elif state == "inconsistent":
        export_annotations(documents, settings.ner_index_path)
        payload = json.loads(settings.ner_index_path.read_text(encoding="utf-8"))
        payload["entity_index"] = {}
        settings.ner_index_path.write_text(json.dumps(payload), encoding="utf-8")
    response = http.post("/api/ner/search", json={"text": "محمد"})
    assert response.status_code == 503
    assert response.json() == {"detail": "Index NER indisponible"}

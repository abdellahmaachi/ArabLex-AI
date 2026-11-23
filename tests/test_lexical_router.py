"""Contrat HTTP et chargement des constructions lexicales publiées."""

import json
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings, get_settings
from backend.app.main import app
from backend.app.routers.lexical import _load_service, get_lexical_service
from src.lexical_search.preprocessing import PreprocessingConfig
from src.lexical_search.service import LexicalSearchService


@pytest.fixture
def client(tmp_path):
    settings = Settings(_env_file=None, artifacts_dir=tmp_path, lexical_top_k=3)
    app.dependency_overrides[get_settings] = lambda: settings
    _load_service.cache_clear()
    with TestClient(app) as client:
        yield client, settings
    app.dependency_overrides.clear()
    _load_service.cache_clear()


def publish(settings, directory):
    service = LexicalSearchService(PreprocessingConfig(stemming=False)).fit([
        {"id": "1", "class": "civil", "filename": "1.txt", "path": "civil/1.txt",
         "text": "عقد ملكية"},
        {"id": "2", "class": "family", "filename": "2.txt", "path": "family/2.txt",
         "text": "إرث زواج"},
    ])
    service.save_artifacts(settings.lexical_index_path.parent / directory)
    settings.lexical_index_path.write_text(
        json.dumps({"version": 1, "directory": directory}), encoding="utf-8",
    )


def test_search_published_index_and_reload(client):
    http, settings = client
    publish(settings, "build-one")
    response = http.post("/api/search/lexical", json={"query": "عقد"})
    assert response.status_code == 200
    assert response.json()["results"][0]["class"] == "civil"
    first = get_lexical_service(settings)
    assert get_lexical_service(settings) is first
    publish(settings, "build-two")
    assert get_lexical_service(settings) is not first
    assert http.post("/api/search/lexical", json={"query": "عقد", "k": 0}).json()["results"] == []
    assert http.post("/api/search/lexical", json={
        "query": "عقد", "class_name": "unknown",
    }).status_code == 422


def test_forwards_options_and_configured_default(client):
    http, _ = client
    service = Mock()
    service.search.return_value = {
        "query": "عقد", "tokens": [], "selected_class": None,
        "class_scores": {}, "results": [],
    }
    app.dependency_overrides[get_lexical_service] = lambda: service
    assert http.post("/api/search/lexical", json={"query": "عقد"}).status_code == 200
    service.search.assert_called_with("عقد", k=3, min_score=0.0, class_name=None)
    assert http.post("/api/search/lexical", json={
        "query": "عقد", "k": 2, "min_score": 0.4, "class_name": "civil",
    }).status_code == 200
    service.search.assert_called_with("عقد", k=2, min_score=0.4, class_name="civil")


@pytest.mark.parametrize("payload", [
    {}, {"query": " "}, {"query": "عقد", "k": -1},
    {"query": "عقد", "k": True}, {"query": "عقد", "min_score": 2},
])
def test_invalid_request_does_not_search(client, payload):
    http, _ = client
    service = Mock()
    app.dependency_overrides[get_lexical_service] = lambda: service
    assert http.post("/api/search/lexical", json=payload).status_code == 422
    service.search.assert_not_called()


@pytest.mark.parametrize("pointer", [None, "not-json", '{"version": 2}',
                                       '{"version": 1, "directory": "../outside"}',
                                       '{"version": 1, "directory": "missing-build"}'])
def test_unavailable_index(client, pointer):
    http, settings = client
    if pointer is not None:
        settings.lexical_index_path.parent.mkdir(parents=True)
        settings.lexical_index_path.write_text(pointer, encoding="utf-8")
    response = http.post("/api/search/lexical", json={"query": "عقد"})
    assert response.status_code == 503
    assert response.json() == {"detail": "Index lexical indisponible"}

"""Contrat de configuration du backend, sans modèles ni services externes."""

import pytest
from pydantic import ValidationError

from backend.app.core.config import PROJECT_ROOT, Settings, get_settings


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_paths_are_independent_of_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    settings = Settings(_env_file=None, artifacts_dir="custom-artifacts")
    assert settings.corpus_dir == PROJECT_ROOT / "data/raw"
    assert settings.lexical_index_path == PROJECT_ROOT / "custom-artifacts/lexical_search/current.json"
    assert settings.qdrant_path == PROJECT_ROOT / "custom-artifacts/rag/qdrant"
    assert settings.ner_index_path == PROJECT_ROOT / "custom-artifacts/ner/entity_index/index.json"
    assert settings.classification_model_path is None
    assert not (PROJECT_ROOT / "custom-artifacts").exists()


def test_source_precedence(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("RAG_TOP_K=7\nGEMINI_MODEL=local-test\nUNRELATED=value\n", encoding="utf-8")
    monkeypatch.setenv("RAG_TOP_K", "9")
    assert Settings(_env_file=env_file).rag_top_k == 9
    assert Settings(_env_file=env_file, rag_top_k=2).rag_top_k == 2
    assert Settings(_env_file=env_file).gemini_model == "local-test"
    monkeypatch.delenv("RAG_TOP_K")
    assert Settings(_env_file=env_file).rag_top_k == 7


@pytest.mark.parametrize("value", ["http://a, http://b", '["http://a", "http://b"]'])
def test_cors_formats(monkeypatch, value):
    monkeypatch.setenv("CORS_ORIGINS", value)
    assert Settings(_env_file=None).cors_origins == ("http://a", "http://b")


def test_gemini_is_optional_and_secret_is_masked():
    settings = Settings(_env_file=None, gemini_api_key="", gemini_model=" ")
    assert settings.gemini_api_key is None
    assert settings.gemini_model is None
    settings = Settings(_env_file=None, gemini_api_key="test-secret")
    assert "test-secret" not in repr(settings)
    assert "test-secret" not in settings.model_dump_json()
    assert settings.gemini_api_key.get_secret_value() == "test-secret"


@pytest.mark.parametrize("options", [
    {"rag_top_k": 0}, {"gemini_timeout_ms": -1}, {"gemini_temperature": 3},
    {"gemini_temperature": float("nan")}, {"log_level": "invalid"},
])
def test_invalid_parameters(options):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **options)


def test_cached_settings(monkeypatch):
    monkeypatch.setattr(Settings, "model_config", {**Settings.model_config, "env_file": None})
    first = get_settings()
    monkeypatch.setenv("RAG_TOP_K", "8")
    assert get_settings() is first
    get_settings.cache_clear()
    assert get_settings().rag_top_k == 8

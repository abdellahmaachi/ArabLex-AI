"""Contrat HTTP de classification et chargement du prédicteur sauvegardé."""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings, get_settings
from backend.app.main import app
from backend.app.routers.classification import _load_predictor, get_classification_predictor
from src.classification.naive_bayes import MultinomialNaiveBayes
from src.classification.predictor import NaiveBayesPredictor
from src.lexical_search.preprocessing import PreprocessingConfig


@pytest.fixture
def client(tmp_path):
    settings = Settings(
        _env_file=None, classification_model_path=tmp_path / "predictor.joblib",
    )
    app.dependency_overrides[get_settings] = lambda: settings
    _load_predictor.cache_clear()
    try:
        with TestClient(app) as http:
            yield http, settings
    finally:
        app.dependency_overrides.clear()
        _load_predictor.cache_clear()


@pytest.mark.parametrize("labels", [["civil", "family"], [10, 20]])
def test_classifies_with_saved_model_without_training(client, monkeypatch, labels):
    http, settings = client
    model = MultinomialNaiveBayes().fit([[3, 0], [0, 3]], labels)
    predictor = NaiveBayesPredictor(
        model, {"عقد": 0, "ارث": 1}, config=PreprocessingConfig(stemming=False),
    )
    predictor.save(settings.classification_model_path)
    forbidden_fit = Mock(side_effect=AssertionError("Aucun entraînement attendu"))
    monkeypatch.setattr(MultinomialNaiveBayes, "fit", forbidden_fit)
    for text, label in zip(["عَقْد", "إرث"], labels):
        response = http.post("/api/classify", json={"text": text})
        assert response.status_code == 200
        assert response.json()["label"] == label
        assert response.json()["probabilities"] == pytest.approx({
            str(key): value for key, value in predictor.predict_proba(text).items()
        })
    assert get_classification_predictor(settings) is get_classification_predictor(settings)
    forbidden_fit.assert_not_called()


def test_preserves_original_text(client):
    http, _ = client
    predictor = Mock()
    predictor.predict.return_value = "civil"
    predictor.predict_proba.return_value = {"civil": 1.0}
    app.dependency_overrides[get_classification_predictor] = lambda: predictor
    text = "  عَقْد\n"
    assert http.post("/api/classify", json={"text": text}).status_code == 200
    predictor.predict.assert_called_once_with(text)
    predictor.predict_proba.assert_called_once_with(text)


@pytest.mark.parametrize("payload", [{}, {"text": ""}, {"text": " \n"},
                                     {"text": None}, {"text": 123},
                                     {"text": "عقد", "unknown": True}])
def test_invalid_request_does_not_predict(client, payload):
    http, _ = client
    predictor = Mock()
    app.dependency_overrides[get_classification_predictor] = lambda: predictor
    assert http.post("/api/classify", json=payload).status_code == 422
    predictor.predict.assert_not_called()
    predictor.predict_proba.assert_not_called()


@pytest.mark.parametrize("state", ["unconfigured", "missing", "corrupt"])
def test_unavailable_model_returns_503(client, state):
    http, settings = client
    if state == "unconfigured":
        settings = Settings(_env_file=None, classification_model_path=None)
        app.dependency_overrides[get_settings] = lambda: settings
    elif state == "corrupt":
        settings.classification_model_path.write_bytes(b"invalid model")
    response = http.post("/api/classify", json={"text": "عقد"})
    assert response.status_code == 503
    assert response.json() == {"detail": "Modèle de classification indisponible"}

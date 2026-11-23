"""Rechargement du modèle et classification de textes avec vocabulaire figé."""

import joblib
import numpy as np
import pytest

from src.classification.naive_bayes import MultinomialNaiveBayes
from src.classification.predictor import NaiveBayesPredictor
from src.common.arabic_normalizer import NormalizationConfig
from src.lexical_search.preprocessing import PreprocessingConfig


@pytest.fixture
def predictor():
    model = MultinomialNaiveBayes().fit(
        [[3, 0], [1, 0], [0, 4]], ["real_estate", "real_estate", "family_inheritance"]
    )
    # L'ordre d'insertion des termes diffère de celui des colonnes.
    return NaiveBayesPredictor(
        model, {"ارث": 1, "عقد": 0}, config=PreprocessingConfig(stemming=False)
    )


def test_predicts_labels_and_counts_in_saved_column_order(predictor):
    assert predictor.predict("عَقْد عقد") == "real_estate"
    assert predictor.predict("إِرْث") == "family_inheritance"
    matrix = predictor.vectorize(["عقد عقد إرث مجهول", "", "في"])
    assert matrix.format == "csr"
    np.testing.assert_array_equal(matrix.toarray(), [[2, 1], [0, 0], [0, 0]])
    assert predictor.predict_many(iter(["عقد", "إرث"])) == ["real_estate", "family_inheritance"]
    assert predictor.predict_many([]) == []


def test_unknown_terms_use_priors_without_changing_vocabulary(predictor):
    before = predictor.vocabulary_
    for text in ("", "في من", "مجهول"):
        assert predictor.predict(text) == "real_estate"
        assert predictor.predict_proba(text) == pytest.approx({
            "real_estate": 2 / 3, "family_inheritance": 1 / 3,
        })
    assert predictor.vocabulary_ == before
    before.clear()
    assert len(predictor.vocabulary_) == 2


def test_roundtrip_requires_no_training(predictor, tmp_path, monkeypatch):
    expected = predictor.predict_proba("عقد إرث")
    path = tmp_path / "model" / "predictor.joblib"
    predictor.save(path)

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("Aucun réentraînement attendu")

    monkeypatch.setattr(MultinomialNaiveBayes, "fit", forbidden_fit)
    loaded = NaiveBayesPredictor.load(path)
    assert loaded.predict_proba("عقد إرث") == pytest.approx(expected)
    assert loaded.predict("إرث") == "family_inheritance"
    assert loaded.vocabulary_ == predictor.vocabulary_


def test_roundtrip_preserves_custom_preprocessing_and_numeric_labels(tmp_path):
    model = MultinomialNaiveBayes().fit([[3, 0], [0, 3]], [10, 20])
    predictor = NaiveBayesPredictor(
        model, {"علي": 0, "عقد": 1},
        config=PreprocessingConfig(
            stemming=False, normalization=NormalizationConfig(normalize_alef_maqsura=True)
        ),
        stopwords=iter(["عقد"]),
    )
    path = tmp_path / "predictor.joblib"
    predictor.save(path)
    loaded = NaiveBayesPredictor.load(path)
    np.testing.assert_array_equal(loaded.vectorize(["عَلَى عقد"]).toarray(), [[1, 0]])
    assert loaded.predict("على") == 10
    assert type(loaded.predict("على")) is int
    assert set(loaded.predict_proba("على")) == {10, 20}


@pytest.mark.parametrize("vocabulary", [{}, {"عقد": 0}, {"عقد": 0, "ارث": 0},
                                         {"عقد": 0, "ارث": 2}, {"عقد": 0, "ارث": 1.0}])
def test_rejects_invalid_or_mismatched_vocabulary(vocabulary):
    model = MultinomialNaiveBayes().fit([[1, 0]], ["a"])
    with pytest.raises(ValueError):
        NaiveBayesPredictor(model, vocabulary)


def test_invalid_model_and_text(predictor):
    with pytest.raises(ValueError, match="entraîné"):
        NaiveBayesPredictor(MultinomialNaiveBayes(), {"عقد": 0})
    with pytest.raises(TypeError):
        predictor.predict(None)
    with pytest.raises(TypeError):
        predictor.predict_many("عقد")


@pytest.mark.parametrize("payload", [None, {"version": 99}, {"version": 1, "model": None}])
def test_rejects_invalid_artifact(payload, tmp_path):
    path = tmp_path / "invalid.joblib"
    joblib.dump(payload, path)
    with pytest.raises(ValueError):
        NaiveBayesPredictor.load(path)


def test_missing_model_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        NaiveBayesPredictor.load(tmp_path / "missing.joblib")

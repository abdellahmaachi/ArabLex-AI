"""Métriques calculées à la main et indépendance des plis de validation."""

import json

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

from src.classification.evaluation import (
    classification_metrics,
    confusion_matrix,
    stratified_cross_validation,
)
from src.classification.naive_bayes import MultinomialNaiveBayes


def test_metrics_match_manual_counts():
    actual = ["a", "a", "a", "b", "b", "c"]
    predicted = ["a", "a", "b", "a", "b", "b"]
    report = classification_metrics(actual, predicted)
    assert report["confusion_matrix"] == [[2, 1, 0], [1, 1, 0], [0, 1, 0]]
    assert report["labels"] == ["a", "b", "c"]
    np.testing.assert_allclose([row["precision"] for row in report["per_class"]], [2/3, 1/3, 0])
    np.testing.assert_allclose([row["recall"] for row in report["per_class"]], [2/3, 1/2, 0])
    np.testing.assert_allclose([row["f1"] for row in report["per_class"]], [2/3, 2/5, 0])
    assert report["macro_avg"]["f1"] == pytest.approx(16/45)
    assert report["weighted_avg"]["f1"] == pytest.approx(7/15)
    assert report["accuracy"] == 0.5
    assert report["micro_avg"] == {"precision": 0.5, "recall": 0.5, "f1": 0.5}
    json.dumps(report, allow_nan=False)


def test_explicit_label_order_and_absent_class():
    matrix = confusion_matrix([1, 1, 2], [1, 2, 2], labels=[2, 1, 3])
    np.testing.assert_array_equal(matrix, [[1, 0, 0], [1, 1, 0], [0, 0, 0]])
    report = classification_metrics([1], [1], labels=[2, 1])
    assert report["per_class"][0] == {
        "label": 2, "support": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
    }
    assert report["macro_avg"]["f1"] == 0.5
    assert report["weighted_avg"]["f1"] == 1


@pytest.mark.parametrize("sparse", [False, True])
def test_stratified_cv_coverage_reproducibility_and_metrics(sparse):
    X = np.array([[3, 0]] * 6 + [[0, 4]] * 3)
    y = ["a"] * 6 + ["b"] * 3
    if sparse:
        X = csr_matrix(X)
    report = stratified_cross_validation(X, iter(y), n_splits=3)
    assert report == stratified_cross_validation(X, y, n_splits=3)
    assert report["predictions"] == y
    assert report["metrics"]["confusion_matrix"] == [[6, 0], [0, 3]]
    assert report["mean_scores"] == {"accuracy": 1, "precision": 1, "recall": 1, "f1": 1}
    assert all(value == 0 for value in report["std_scores"].values())
    all_tests = []
    for fold in report["folds"]:
        train, test = fold["train_indices"], fold["test_indices"]
        assert not set(train) & set(test)
        assert sorted(train + test) == list(range(9))
        assert [y[index] for index in test].count("a") == 2
        assert [y[index] for index in test].count("b") == 1
        all_tests.extend(test)
    assert sorted(all_tests) == list(range(9))
    json.dumps(report, allow_nan=False)


def test_each_fold_fits_a_new_model_on_training_rows_only(monkeypatch):
    original_fit = MultinomialNaiveBayes.fit
    fitted = []

    def tracked_fit(self, X, y):
        fitted.append((self, np.asarray(X).copy()))
        return original_fit(self, X, y)

    monkeypatch.setattr(MultinomialNaiveBayes, "fit", tracked_fit)
    X = np.array([[1, 0], [2, 0], [3, 0], [0, 4], [0, 5], [0, 6]])
    report = stratified_cross_validation(X, [0, 0, 0, 1, 1, 1], n_splits=3, shuffle=False)
    assert len({id(model) for model, _ in fitted}) == 3
    for (_, actual_rows), fold in zip(fitted, report["folds"]):
        np.testing.assert_array_equal(actual_rows, X[fold["train_indices"]])
    assert report["random_state"] is None


@pytest.mark.parametrize(
    ("actual", "predicted", "labels"),
    [([], [], None), ([1], [], None), ([1], [2], [1]), ([1], [1], [1, 1]),
     ([1], ["a"], None), ([float("nan")], [1], None)],
)
def test_invalid_metric_inputs(actual, predicted, labels):
    with pytest.raises(ValueError):
        classification_metrics(actual, predicted, labels=labels)


@pytest.mark.parametrize("splits", [0, 1, True, 2.5, 4])
def test_invalid_fold_count(splits):
    with pytest.raises(ValueError):
        stratified_cross_validation([[1]] * 6, [0, 0, 0, 1, 1, 1], n_splits=splits)


def test_invalid_cv_inputs():
    with pytest.raises(ValueError, match="deux classes"):
        stratified_cross_validation([[1]] * 4, [0] * 4, n_splits=2)
    with pytest.raises(ValueError, match="étiquette"):
        stratified_cross_validation([[1]] * 4, [0, 1], n_splits=2)
    with pytest.raises(ValueError):
        stratified_cross_validation([[-1]] * 4, [0, 0, 1, 1], n_splits=2)


def test_text_cross_validation_learns_vocabulary_only_from_training_fold():
    texts = [f"shared unique{index}" for index in range(6)]
    vectorizers = []

    def factory():
        vectorizer = CountVectorizer()
        vectorizers.append(vectorizer)
        return vectorizer

    report = stratified_cross_validation(
        texts, [0, 0, 0, 1, 1, 1], n_splits=3, vectorizer_factory=factory,
    )
    assert len(vectorizers) == 3
    for vectorizer, fold in zip(vectorizers, report["folds"]):
        expected = {"shared"} | {f"unique{index}" for index in fold["train_indices"]}
        assert set(vectorizer.vocabulary_) == expected
        assert all(f"unique{index}" not in vectorizer.vocabulary_ for index in fold["test_indices"])

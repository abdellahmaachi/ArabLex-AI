"""Probabilités, lissage, prédictions et évaluation du Naive Bayes maison."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.classification.evaluation import (
    classification_metrics,
    confusion_matrix,
    stratified_cross_validation,
)
from src.classification.naive_bayes import MultinomialNaiveBayes


@pytest.fixture
def training_data():
    return np.array([[2, 1, 0], [1, 0, 0], [0, 1, 3]]), ["a", "a", "b"]


@pytest.mark.parametrize("sparse", [False, True])
def test_counts_laplace_and_log_probabilities(training_data, sparse):
    X, y = training_data
    if sparse:
        X = csr_matrix(X)
    original = X.copy()
    model = MultinomialNaiveBayes().fit(X, iter(y))

    np.testing.assert_array_equal(model.classes_, ["a", "b"])
    np.testing.assert_array_equal(model.class_count_, [2, 1])
    np.testing.assert_array_equal(model.feature_count_, [[3, 1, 0], [0, 1, 3]])
    np.testing.assert_allclose(np.exp(model.class_log_prior_), [2 / 3, 1 / 3])
    np.testing.assert_allclose(
        np.exp(model.feature_log_prob_), [[4 / 7, 2 / 7, 1 / 7], [1 / 7, 2 / 7, 4 / 7]]
    )
    if sparse:
        assert (X != original).nnz == 0
    else:
        np.testing.assert_array_equal(X, original)


def test_predictions_match_hand_calculated_posteriors(training_data):
    model = MultinomialNaiveBayes().fit(*training_data)
    queries = [[1, 0, 0], [0, 0, 1], [0, 0, 0]]
    expected = [[8 / 9, 1 / 9], [1 / 3, 2 / 3], [2 / 3, 1 / 3]]
    np.testing.assert_array_equal(model.predict(queries), ["a", "b", "a"])
    np.testing.assert_allclose(model.predict_proba(queries), expected)
    np.testing.assert_allclose(model.predict_log_proba(queries), np.log(expected))
    np.testing.assert_allclose(model.predict_proba(csr_matrix(queries)), expected)


def test_custom_smoothing_and_unseen_feature():
    model = MultinomialNaiveBayes(alpha=0.5).fit([[2, 0], [0, 0]], [0, 1])
    np.testing.assert_allclose(np.exp(model.feature_log_prob_), [[2.5 / 3, 0.5 / 3], [0.5, 0.5]])
    assert np.isfinite(model.predict_log_proba([[0, 1]])).all()


def test_large_documents_use_stable_log_probabilities(training_data):
    model = MultinomialNaiveBayes().fit(*training_data)
    probabilities = model.predict_proba([[100000, 0, 0], [0, 0, 100000]])
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), [1, 1])
    np.testing.assert_array_equal(model.predict([[100000, 0, 0], [0, 0, 100000]]), ["a", "b"])
    assert np.isfinite(model.predict_log_proba([[100000, 0, 0]])).all()


def test_zero_counts_ties_single_class_and_empty_predictions():
    model = MultinomialNaiveBayes().fit(np.zeros((2, 3)), ["z", "a"])
    np.testing.assert_allclose(np.exp(model.feature_log_prob_), np.full((2, 3), 1 / 3))
    assert model.predict([[0, 0, 0]]).tolist() == ["a"]
    assert model.predict(np.empty((0, 3))).shape == (0,)
    assert model.predict_proba(csr_matrix((0, 3))).shape == (0, 2)
    single = MultinomialNaiveBayes().fit([[0, 0]], ["only"])
    np.testing.assert_allclose(single.predict_proba([[1, 0]]), [[1]])


def test_refit_replaces_counts_and_failed_fit_preserves_model(training_data):
    model = MultinomialNaiveBayes().fit(*training_data)
    with pytest.raises(ValueError):
        model.fit([[-1, 0, 0]], ["bad"])
    np.testing.assert_array_equal(model.class_count_, [2, 1])
    model.fit([[1, 0], [0, 2]], [2, 1])
    np.testing.assert_array_equal(model.classes_, [1, 2])
    np.testing.assert_array_equal(model.feature_count_, [[0, 2], [1, 0]])
    assert model.n_features_in_ == 2


@pytest.mark.parametrize("alpha", [0, -1, float("nan"), float("inf")])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        MultinomialNaiveBayes(alpha)


@pytest.mark.parametrize("alpha", [True, "1", None])
def test_non_numeric_alpha(alpha):
    with pytest.raises(TypeError):
        MultinomialNaiveBayes(alpha)


@pytest.mark.parametrize(
    ("X", "y"),
    [
        (np.empty((0, 2)), []),
        (np.empty((1, 0)), ["a"]),
        ([1, 2], ["a"]),
        ([[-1]], ["a"]),
        (csr_matrix([[-1]]), ["a"]),
        ([[float("nan")]], ["a"]),
        ([[float("inf")]], ["a"]),
        ([[1j]], ["a"]),
        ([[1], [2]], ["a"]),
        ([[1], [2]], [1, "a"]),
        ([[1]], [float("nan")]),
        ([[1]], [["a"]]),
    ],
)
def test_invalid_training_data(X, y):
    with pytest.raises(ValueError):
        MultinomialNaiveBayes().fit(X, y)


def test_prediction_validation(training_data):
    model = MultinomialNaiveBayes()
    with pytest.raises(RuntimeError, match="fit"):
        model.predict([[1, 0, 0]])
    model.fit(*training_data)
    with pytest.raises(ValueError, match="colonnes"):
        model.predict([[1, 0]])
    with pytest.raises(ValueError):
        model.predict([[-1, 0, 0]])


@pytest.mark.parametrize("alpha", [0.25, 1.0, 5.0])
def test_smoothing_denominator_includes_all_vocabulary_columns(alpha):
    # La troisième colonne n'est observée dans aucune classe, mais compte dans V.
    model = MultinomialNaiveBayes(alpha).fit([[3, 1, 0], [0, 2, 0]], ["a", "b"])
    expected = np.array([
        [(3 + alpha) / (4 + 3 * alpha), (1 + alpha) / (4 + 3 * alpha),
         alpha / (4 + 3 * alpha)],
        [alpha / (2 + 3 * alpha), (2 + alpha) / (2 + 3 * alpha),
         alpha / (2 + 3 * alpha)],
    ])
    actual = np.exp(model.feature_log_prob_)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(actual.sum(axis=1), [1, 1])
    assert np.all(actual > 0)
    assert np.isfinite(model.predict_log_proba([[0, 0, 1]])).all()


def test_multiclass_posteriors_include_priors_and_term_multiplicity():
    model = MultinomialNaiveBayes().fit(
        [[0, 2], [2, 0], [0, 2], [1, 1]], ["c", "a", "c", "b"]
    )
    # Classes triées a, b, c ; a priori 1/4, 1/4, 1/2.
    # Pour x=(2,1), les probabilités jointes sont calculées directement.
    joint = np.array([
        (1 / 4) * (3 / 4)**2 * (1 / 4),
        (1 / 4) * (1 / 2)**2 * (1 / 2),
        (1 / 2) * (1 / 6)**2 * (5 / 6),
    ])
    expected = joint / joint.sum()
    np.testing.assert_array_equal(model.classes_, ["a", "b", "c"])
    np.testing.assert_allclose(model.predict_proba([[2, 1]])[0], expected)
    np.testing.assert_allclose(model.predict_log_proba([[2, 1]])[0], np.log(expected))
    assert model.predict([[2, 1]]).tolist() == ["a"]
    np.testing.assert_allclose(model.predict_proba([[0, 0]])[0], [1 / 4, 1 / 4, 1 / 2])


@pytest.mark.parametrize("alpha", [1e-12, 1.0, 1e100])
def test_probabilities_are_normalized_for_extreme_positive_smoothing(training_data, alpha):
    model = MultinomialNaiveBayes(alpha).fit(*training_data)
    queries = csr_matrix([[0, 0, 0], [4, 2, 0], [0, 1, 8]])
    probabilities = model.predict_proba(queries)
    log_probabilities = model.predict_log_proba(queries)
    assert np.isfinite(log_probabilities).all()
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(3))
    np.testing.assert_allclose(np.exp(log_probabilities), probabilities)
    np.testing.assert_array_equal(
        model.predict(queries), model.classes_[np.argmax(probabilities, axis=1)]
    )


def test_predictions_are_invariant_to_consistent_column_permutation(training_data):
    X, y = training_data
    queries = np.array([[3, 0, 1], [0, 0, 3], [0, 0, 0]])
    permutation = [2, 0, 1]
    original = MultinomialNaiveBayes().fit(X, y)
    permuted = MultinomialNaiveBayes().fit(csr_matrix(X[:, permutation]), y)
    np.testing.assert_allclose(
        original.predict_proba(queries), permuted.predict_proba(queries[:, permutation])
    )
    np.testing.assert_array_equal(
        original.predict(queries), permuted.predict(queries[:, permutation])
    )


def test_evaluation_of_model_predictions_matches_manual_confusion(training_data):
    model = MultinomialNaiveBayes().fit(*training_data)
    predicted = model.predict([[1, 0, 0], [0, 0, 1], [0, 0, 0], [0, 1, 0]])
    actual = ["a", "b", "b", "b"]
    report = classification_metrics(actual, predicted)

    assert predicted.tolist() == ["a", "b", "a", "a"]
    np.testing.assert_array_equal(confusion_matrix(actual, predicted), [[1, 0], [2, 1]])
    assert report["confusion_matrix"] == [[1, 0], [2, 1]]
    np.testing.assert_allclose([row["precision"] for row in report["per_class"]], [1/3, 1])
    np.testing.assert_allclose([row["recall"] for row in report["per_class"]], [1, 1/3])
    np.testing.assert_allclose([row["f1"] for row in report["per_class"]], [0.5, 0.5])
    assert [row["support"] for row in report["per_class"]] == [1, 3]
    assert report["accuracy"] == 0.5
    assert report["macro_avg"]["precision"] == pytest.approx(2/3)
    assert report["weighted_avg"]["precision"] == pytest.approx(5/6)
    assert report["weighted_avg"]["recall"] == pytest.approx(0.5)
    assert report["macro_avg"]["f1"] == pytest.approx(0.5)


def test_evaluation_handles_classes_never_predicted():
    report = classification_metrics(["b", "b"], ["a", "a"], labels=["b", "a", "c"])
    assert report["confusion_matrix"] == [[0, 2, 0], [0, 0, 0], [0, 0, 0]]
    for row in report["per_class"]:
        assert row["precision"] == row["recall"] == row["f1"] == 0.0
    assert report["accuracy"] == 0.0
    assert report["macro_avg"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


@pytest.mark.parametrize("sparse", [False, True])
def test_cross_validation_reports_non_perfect_out_of_fold_predictions(sparse):
    # Caractéristiques identiques : impossible de distinguer les classes équilibrées.
    X = np.array([[1, 0]] * 8)
    if sparse:
        X = csr_matrix(X)
    labels = ["a"] * 4 + ["b"] * 4
    report = stratified_cross_validation(X, labels, n_splits=2, random_state=17)

    assert report == stratified_cross_validation(X, labels, n_splits=2, random_state=17)
    assert report["predictions"] == ["a"] * 8
    assert report["metrics"]["confusion_matrix"] == [[4, 0], [4, 0]]
    assert report["metrics"]["accuracy"] == 0.5
    assert report["mean_scores"]["f1"] == pytest.approx(1/3)
    assert report["std_scores"]["f1"] == 0.0
    accumulated_confusion = np.zeros((2, 2), dtype=int)
    test_indices = []
    for fold in report["folds"]:
        train, test = fold["train_indices"], fold["test_indices"]
        assert set(train).isdisjoint(test)
        assert [labels[index] for index in test].count("a") == 2
        assert [labels[index] for index in test].count("b") == 2
        accumulated_confusion += np.asarray(fold["metrics"]["confusion_matrix"])
        test_indices.extend(test)
    assert sorted(test_indices) == list(range(8))
    np.testing.assert_array_equal(accumulated_confusion, report["metrics"]["confusion_matrix"])

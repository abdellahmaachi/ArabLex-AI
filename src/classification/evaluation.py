"""Métriques multiclasse et validation croisée du Naive Bayes maison."""

from collections import Counter
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import spmatrix
from sklearn.model_selection import StratifiedKFold

from src.classification.naive_bayes import MultinomialNaiveBayes, _validate_features

Label = str | int


def _labels(values: Iterable[Label]) -> list[Label]:
    if isinstance(values, (str, bytes)):
        raise ValueError("Les étiquettes doivent être une séquence")
    items = list(values)
    if all(isinstance(item, (str, np.str_)) for item in items):
        return [str(item) for item in items]
    if all(isinstance(item, (int, np.integer)) for item in items):
        return [int(item) for item in items]
    raise ValueError("Les étiquettes doivent être toutes des chaînes ou toutes des entiers")


def _confusion_data(
    y_true: Iterable[Label], y_pred: Iterable[Label], labels: Iterable[Label] | None
) -> tuple[list[Label], np.ndarray]:
    actual, predicted = _labels(y_true), _labels(y_pred)
    if not actual or len(actual) != len(predicted):
        raise ValueError("y_true et y_pred doivent être non vides et de même longueur")
    observed = _labels(actual + predicted)
    classes = sorted(set(observed)) if labels is None else _labels(labels)
    _labels(observed + classes)
    if len(set(classes)) != len(classes):
        raise ValueError("labels ne doit pas contenir de doublons")
    if not set(observed).issubset(classes):
        raise ValueError("labels doit inclure toutes les classes observées et prédites")
    positions = {label: index for index, label in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for actual_label, predicted_label in zip(actual, predicted):
        matrix[positions[actual_label], positions[predicted_label]] += 1
    return classes, matrix


def confusion_matrix(
    y_true: Iterable[Label],
    y_pred: Iterable[Label],
    *,
    labels: Iterable[Label] | None = None,
) -> np.ndarray:
    """Matrice : lignes = classes réelles, colonnes = classes prédites.

    L'ordre est celui de ``labels`` ou l'ordre trié des classes observées/prédites.
    Des classes absentes peuvent être indiquées explicitement (lignes/colonnes nulles).
    """
    return _confusion_data(y_true, y_pred, labels)[1]


def classification_metrics(
    y_true: Iterable[Label],
    y_pred: Iterable[Label],
    *,
    labels: Iterable[Label] | None = None,
) -> dict[str, Any]:
    """Précision, rappel, F1, support, accuracy et matrice de confusion.

    Précision = TP/(TP+FP), rappel = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN).
    Toute division par zéro donne 0. La moyenne macro donne le même poids aux
    classes ; la moyenne pondérée utilise leur support réel. En classification
    à une seule étiquette, précision/rappel/F1 micro sont égaux à l'accuracy.
    Le résultat utilise uniquement des types sérialisables en JSON.
    """
    classes, matrix = _confusion_data(y_true, y_pred, labels)
    support = matrix.sum(axis=1)
    predicted_count = matrix.sum(axis=0)
    true_positive = np.diag(matrix).astype(float)
    precision = np.divide(
        true_positive, predicted_count,
        out=np.zeros(len(classes)), where=predicted_count != 0,
    )
    recall = np.divide(
        true_positive, support, out=np.zeros(len(classes)), where=support != 0,
    )
    denominator = support + predicted_count
    f1 = np.divide(
        2 * true_positive, denominator,
        out=np.zeros(len(classes)), where=denominator != 0,
    )
    total = int(support.sum())
    accuracy = float(true_positive.sum() / total)
    arrays = {"precision": precision, "recall": recall, "f1": f1}
    return {
        "labels": classes,
        "confusion_matrix": matrix.tolist(),
        "per_class": [
            {"label": label, "support": int(support[index]),
             **{name: float(values[index]) for name, values in arrays.items()}}
            for index, label in enumerate(classes)
        ],
        "accuracy": accuracy,
        "support": total,
        "macro_avg": {name: float(values.mean()) for name, values in arrays.items()},
        "weighted_avg": {
            name: float(np.dot(values, support) / total) for name, values in arrays.items()
        },
        "micro_avg": {name: accuracy for name in arrays},
    }


def stratified_cross_validation(
    X: ArrayLike | spmatrix,
    y: Iterable[Label],
    *,
    n_splits: int = 5,
    alpha: float = 1.0,
    shuffle: bool = True,
    random_state: int | None = 42,
    vectorizer_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Évalue un nouveau MultinomialNaiveBayes sur chaque pli stratifié.

    Accepte des matrices denses ou creuses. Il faut au moins deux classes et
    ``n_splits`` exemples par classe pour qu'elles soient présentes dans chaque
    pli. Chaque exemple est prédit exactement une fois hors apprentissage.

    ``metrics`` agrège ces prédictions ; ``mean_scores`` et ``std_scores`` sont
    la moyenne et l'écart-type population (ddof=0) des scores par pli. Précision,
    rappel et F1 y utilisent la moyenne macro. Les indices et prédictions sont
    fournis dans l'ordre d'origine. ``random_state`` est ignoré sans mélange.

    X doit représenter des caractéristiques fixes, par exemple des comptages.
    Ne pas fournir des poids IDF ou une sélection Chi² appris sur tout le corpus :
    ces étapes doivent être ajustées uniquement sur l'entraînement de chaque pli.
    Avec ``vectorizer_factory``, X est une collection de textes bruts. La fabrique
    doit renvoyer un nouveau vectoriseur doté de fit_transform/transform ; son
    vocabulaire est appris exclusivement sur l'entraînement de chaque pli.
    """
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits doit être un entier supérieur ou égal à 2")
    if not isinstance(shuffle, bool):
        raise TypeError("shuffle doit être un booléen")
    MultinomialNaiveBayes(alpha=alpha)  # Valider le lissage avant le découpage.
    if vectorizer_factory is None:
        matrix = _validate_features(X)
    else:
        if not callable(vectorizer_factory):
            raise TypeError("vectorizer_factory doit être une fonction de construction")
        if isinstance(X, str):
            raise TypeError("X doit être une collection de textes")
        texts = list(X)
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("X doit contenir uniquement des textes avec vectorizer_factory")
        matrix = np.asarray(texts, dtype=object)
    values = _labels(y)
    if not values or len(values) != matrix.shape[0]:
        raise ValueError("y doit contenir exactement une étiquette par ligne de X non vide")
    counts = Counter(values)
    if len(counts) < 2:
        raise ValueError("La validation stratifiée nécessite au moins deux classes")
    if min(counts.values()) < n_splits:
        raise ValueError("Chaque classe doit contenir au moins n_splits exemples")
    classes = sorted(counts)
    targets = np.asarray(values)
    predictions = np.empty_like(targets)
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=shuffle, random_state=random_state if shuffle else None,
    )
    folds = []
    score_rows = []
    for fold_number, (train, test) in enumerate(splitter.split(matrix, targets), start=1):
        model = MultinomialNaiveBayes(alpha=alpha)
        if vectorizer_factory is None:
            train_features, test_features = matrix[train], matrix[test]
        else:
            vectorizer = vectorizer_factory()
            train_features = vectorizer.fit_transform(matrix[train])
            test_features = vectorizer.transform(matrix[test])
        model.fit(train_features, targets[train])
        predicted = model.predict(test_features)
        predictions[test] = predicted
        metrics = classification_metrics(targets[test], predicted, labels=classes)
        folds.append({
            "fold": fold_number,
            "train_indices": train.tolist(),
            "test_indices": test.tolist(),
            "metrics": metrics,
        })
        score_rows.append({"accuracy": metrics["accuracy"], **metrics["macro_avg"]})
    return {
        "n_splits": n_splits,
        "alpha": float(alpha),
        "shuffle": shuffle,
        "random_state": random_state if shuffle else None,
        "y_true": values,
        "predictions": predictions.tolist(),
        "folds": folds,
        "metrics": classification_metrics(values, predictions, labels=classes),
        "mean_scores": {
            name: float(np.mean([row[name] for row in score_rows])) for name in score_rows[0]
        },
        "std_scores": {
            name: float(np.std([row[name] for row in score_rows])) for name in score_rows[0]
        },
    }

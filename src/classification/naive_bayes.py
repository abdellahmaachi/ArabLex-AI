"""Multinomial Naive Bayes implémenté avec NumPy/SciPy, sans classifieur externe."""

import math
from collections.abc import Iterable
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import csr_matrix, issparse, spmatrix


def _validate_alpha(alpha: float) -> None:
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise TypeError("alpha doit être un nombre réel")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha doit être fini et strictement positif")


def _validate_features(values: ArrayLike | spmatrix) -> np.ndarray | csr_matrix:
    """Valide sans densifier une matrice creuse ni modifier les entrées."""
    if issparse(values):
        if values.ndim != 2 or np.issubdtype(values.dtype, np.complexfloating):
            raise ValueError("X doit être une matrice 2D de nombres réels")
        matrix = csr_matrix(values, dtype=np.float64, copy=True)
        data = matrix.data
    else:
        raw = np.asarray(values)
        if raw.ndim != 2 or np.iscomplexobj(raw):
            raise ValueError("X doit être une matrice 2D de nombres réels")
        matrix = np.asarray(raw, dtype=np.float64)
        data = matrix
    if not np.isfinite(data).all() or np.any(data < 0):
        raise ValueError("X doit contenir des valeurs finies et positives ou nulles")
    if matrix.shape[1] == 0:
        raise ValueError("X doit contenir au moins une caractéristique")
    return matrix


def _log_normalize(values: np.ndarray) -> np.ndarray:
    """Normalise chaque ligne en log avec la méthode log-sum-exp stable."""
    shifted = values - np.max(values, axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


class MultinomialNaiveBayes:
    """Classifieur multinomial avec lissage additif et calculs en logarithmes.

    ``X`` a une ligne par document et une colonne par terme ; la vectorisation
    est réalisée en amont. Utiliser le même vocabulaire et le même ordre de
    colonnes pour l'apprentissage et les prédictions. Les comptages sont attendus,
    mais des poids réels non négatifs sont également acceptés.

    N_c = nombre de documents de classe c ; N_ct = somme des comptages du terme t
    dans cette classe ; V = nombre de colonnes :

        P(c) = N_c / N
        P(t | c) = (N_ct + alpha) / (sum_t N_ct + alpha * V)
        log P(c, x) = log P(c) + sum_t x_t * log P(t | c)

    ``alpha=1`` correspond au lissage de Laplace. Les classes observées sont
    triées ; une égalité de scores favorise la première classe. Une ligne nulle
    utilise uniquement les a priori. Les classes sont des chaînes ou des entiers
    homogènes ; une seule classe est autorisée.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        _validate_alpha(alpha)
        self.alpha = alpha
        self._fitted = False

    def fit(self, X: ArrayLike | spmatrix, y: Iterable[str | int]) -> "MultinomialNaiveBayes":
        """Apprend les comptages et log-probabilités ; renvoie le modèle.

        Un nouvel appel remplace les statistiques. En cas d'entrée invalide,
        le modèle précédemment appris reste utilisable.
        """
        _validate_alpha(self.alpha)
        matrix = _validate_features(X)
        if matrix.shape[0] == 0:
            raise ValueError("Le jeu d'apprentissage ne doit pas être vide")
        if isinstance(y, (str, bytes)):
            raise ValueError("y doit être une séquence d'étiquettes")
        labels = list(y)
        if not (
            all(isinstance(label, (str, np.str_)) for label in labels)
            or all(isinstance(label, (int, np.integer)) for label in labels)
        ):
            raise ValueError("Les étiquettes doivent être toutes des chaînes ou toutes des entiers")
        labels = np.asarray(labels)
        if labels.ndim != 1 or len(labels) != matrix.shape[0]:
            raise ValueError("y doit contenir exactement une étiquette par ligne de X")
        classes, encoded = np.unique(labels, return_inverse=True)
        class_count = np.bincount(encoded, minlength=len(classes)).astype(np.float64)
        feature_count = np.empty((len(classes), matrix.shape[1]), dtype=np.float64)
        with np.errstate(over="ignore"):
            for class_index in range(len(classes)):
                feature_count[class_index] = np.asarray(
                    matrix[encoded == class_index].sum(axis=0)
                ).ravel()
        if not np.isfinite(feature_count).all():
            raise ValueError("Les comptages cumulés dépassent la capacité numérique")

        # log(N_ct + alpha) sans addition susceptible de déborder.
        with np.errstate(divide="ignore"):
            smoothed_log_counts = np.logaddexp(np.log(feature_count), math.log(self.alpha))
        feature_log_prob = _log_normalize(smoothed_log_counts)
        class_log_prior = np.log(class_count) - math.log(matrix.shape[0])

        self.classes_ = classes
        self.class_count_ = class_count
        self.feature_count_ = feature_count
        self.feature_log_prob_ = feature_log_prob
        self.class_log_prior_ = class_log_prior
        self.n_features_in_ = matrix.shape[1]
        self._fitted = True
        return self

    def _joint_log_likelihood(self, X: ArrayLike | spmatrix) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Appeler fit avant de prédire")
        matrix = _validate_features(X)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError("X doit avoir le même nombre de colonnes qu'à l'apprentissage")
        with np.errstate(over="ignore", invalid="ignore"):
            scores = np.asarray(matrix @ self.feature_log_prob_.T) + self.class_log_prior_
        if not np.isfinite(scores).all():
            raise ValueError("Les scores dépassent la capacité numérique")
        return scores

    def predict(self, X: ArrayLike | spmatrix) -> np.ndarray:
        """Renvoie la classe de score logarithmique maximal pour chaque document."""
        scores = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_log_proba(self, X: ArrayLike | spmatrix) -> np.ndarray:
        """Renvoie les log-probabilités a posteriori dans l'ordre de ``classes_``."""
        scores = self._joint_log_likelihood(X)
        return _log_normalize(scores)

    def predict_proba(self, X: ArrayLike | spmatrix) -> np.ndarray:
        """Renvoie des probabilités normalisées (une ligne par document)."""
        return np.exp(self.predict_log_proba(X))

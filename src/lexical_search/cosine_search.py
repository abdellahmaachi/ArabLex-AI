"""Similarité cosinus et classement des documents dans l'espace TF-IDF."""

import heapq
import math
from collections.abc import Iterable
from typing import TypedDict

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import csr_matrix, issparse, spmatrix
from sklearn.metrics.pairwise import cosine_similarity


class SearchResult(TypedDict):
    """Identifiant du document et score de similarité cosinus."""

    id: str
    score: float


def _as_matrix(values: ArrayLike | spmatrix, *, query: bool = False) -> csr_matrix:
    if not issparse(values):
        values = np.asarray(values)
        if query and values.ndim == 1:
            values = values.reshape(1, -1)
    if values.ndim != 2:
        raise ValueError("Les vecteurs doivent former une matrice à deux dimensions")
    raw_values = values.data if issparse(values) else values
    if np.iscomplexobj(raw_values):
        raise ValueError("Les vecteurs doivent contenir des nombres réels")
    matrix = csr_matrix(values, dtype=float, copy=True)
    if not np.isfinite(matrix.data).all():
        raise ValueError("Les vecteurs doivent contenir uniquement des valeurs finies")
    return matrix


def cosine_similarities(
    query_vector: ArrayLike | spmatrix,
    document_matrix: ArrayLike | spmatrix,
) -> np.ndarray:
    """Renvoie un score par ligne du corpus, dans le même ordre.

    Accepte des matrices denses ou creuses et une seule requête (vecteur 1D ou
    matrice à une ligne). Les colonnes doivent correspondre au même vocabulaire,
    dans le même ordre. Les vecteurs n'ont pas besoin d'être normalisés au préalable.
    Un vecteur nul reçoit un score nul. Un corpus vide ou sans colonne est accepté.
    Le corpus reste creux pendant le calcul ; seul le tableau des scores est dense.
    Les entrées ne sont pas modifiées.
    """
    query = _as_matrix(query_vector, query=True)
    documents = _as_matrix(document_matrix)
    if query.shape[0] != 1:
        raise ValueError("Une seule requête est attendue")
    if query.shape[1] != documents.shape[1]:
        raise ValueError("La requête et le corpus doivent avoir le même nombre de colonnes")
    if documents.shape[0] == 0 or documents.shape[1] == 0:
        return np.zeros(documents.shape[0], dtype=float)
    scores = cosine_similarity(documents, query).ravel()
    # Corriger les éventuels dépassements dus aux arrondis flottants.
    return np.clip(scores, -1.0, 1.0)


def rank_top_k(
    scores: ArrayLike,
    document_ids: Iterable[str],
    *,
    k: int = 10,
    min_score: float = 0.0,
) -> list[SearchResult]:
    """Classe au plus k documents par score décroissant, puis par identifiant.

    Le seuil est strict : seuls les scores supérieurs à ``min_score`` sont
    retenus. Par défaut, les scores nuls ou négatifs sont exclus. Les identifiants
    doivent être uniques et alignés sur les scores. ``k=0`` renvoie une liste vide.
    Un k supérieur au nombre de candidats renvoie tous les candidats admissibles.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("k doit être un entier positif ou nul")
    if not math.isfinite(min_score) or not -1 <= min_score <= 1:
        raise ValueError("min_score doit être fini et compris entre -1 et 1")
    raw_scores = np.asarray(scores)
    if np.iscomplexobj(raw_scores):
        raise ValueError("Les scores doivent être réels")
    values = np.asarray(raw_scores, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Les scores doivent être un tableau 1D de valeurs finies")
    if np.any((values < -1) | (values > 1)):
        raise ValueError("Les scores cosinus doivent être compris entre -1 et 1")
    if isinstance(document_ids, str):
        raise TypeError("document_ids doit être une collection d'identifiants")
    identifiers = tuple(document_ids)
    if len(identifiers) != len(values):
        raise ValueError("Un identifiant est requis pour chaque score")
    if any(not isinstance(doc_id, str) or not doc_id.strip() for doc_id in identifiers):
        raise ValueError("Les identifiants doivent être des chaînes non vides")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Les identifiants doivent être uniques")

    candidates = (index for index, score in enumerate(values) if score > min_score)
    best = heapq.nsmallest(k, candidates, key=lambda index: (-values[index], identifiers[index]))
    return [{"id": identifiers[index], "score": float(values[index])} for index in best]


def search_top_k(
    query_vector: ArrayLike | spmatrix,
    document_matrix: ArrayLike | spmatrix,
    document_ids: Iterable[str],
    *,
    k: int = 10,
    min_score: float = 0.0,
) -> list[SearchResult]:
    """Calcule les scores et renvoie les meilleurs documents.

    Exemple avec le modèle TF-IDF ::

        query = model.vectorize_query("عقد ملكية")
        results = search_top_k(query, corpus_matrix, model.document_ids_, k=5)

    Une requête vide ou entièrement hors vocabulaire ne produit aucun résultat
    avec le seuil par défaut.
    """
    scores = cosine_similarities(query_vector, document_matrix)
    return rank_top_k(scores, document_ids, k=k, min_score=min_score)

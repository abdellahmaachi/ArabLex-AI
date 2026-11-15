"""Scores Chi² de présence documentaire et sélection de termes discriminants."""

import math
from collections import Counter
from collections.abc import Iterable, Mapping

from src.common.document_schema import Document
from src.lexical_search.indexer import build_inverted_index
from src.lexical_search.preprocessing import DEFAULT_STOPWORDS, PreprocessingConfig


def compute_chi2_scores(
    documents: Iterable[Document],
    config: PreprocessingConfig | None = None,
    *,
    stopwords: Iterable[str] = DEFAULT_STOPWORDS,
) -> dict[str, dict[str, float]]:
    """Renvoie ``{classe: {terme: score}}`` pour chaque classe observée.

    Pour chaque terme et classe, la table de contingence est :
    A = documents de la classe contenant le terme ;
    B = documents hors classe contenant le terme ;
    C = documents de la classe sans le terme ;
    D = documents hors classe sans le terme.

    Chi² = N * (A*D - B*C)² / ((A+B)*(C+D)*(A+C)*(B+D)), sans correction
    de Yates. Les répétitions d'un terme dans un document ne comptent qu'une fois.
    Les documents vides contribuent aux effectifs de leur classe. Un terme
    présent partout reçoit 0 (aucune capacité de discrimination).

    Il faut au moins deux classes non vides. Un corpus sans terme renvoie un
    dictionnaire vide par classe. Les erreurs de documents/prétraitement sont
    propagées ; les textes originaux restent inchangés.

    Chi² mesure aussi les associations négatives : il n'indique pas la classe
    favorisée par un terme. Avec deux classes, les deux scores sont identiques.
    Calculer ces scores sur le jeu d'entraînement uniquement lors d'une évaluation.
    """
    corpus = list(documents)
    if not corpus:
        raise ValueError("Le corpus ne doit pas être vide")
    class_counts: Counter[str] = Counter()
    for document in corpus:
        label = document["class"]
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Chaque document doit avoir une classe non vide de type chaîne")
        class_counts[label] += 1
    if len(class_counts) < 2:
        raise ValueError("Le calcul Chi² nécessite au moins deux classes")

    index = build_inverted_index(corpus, config, stopwords=stopwords)
    labels = {document["id"]: document["class"] for document in corpus}
    scores: dict[str, dict[str, float]] = {label: {} for label in sorted(class_counts)}
    total = len(corpus)
    for term, postings in index.items():
        present_by_class = Counter(labels[document_id] for document_id in postings)
        present = len(postings)
        for label in scores:
            a = present_by_class[label]
            b = present - a
            c = class_counts[label] - a
            d = total - a - b - c
            denominator = (a + b) * (c + d) * (a + c) * (b + d)
            scores[label][term] = (
                total * (a * d - b * c) ** 2 / denominator if denominator else 0.0
            )
    return scores


def select_discriminant_terms(
    scores: Mapping[str, Mapping[str, float]],
    *,
    k: int | None = 100,
    min_score: float = 0.0,
) -> dict[str, list[str]]:
    """Sélectionne au plus ``k`` termes par classe, par score décroissant.

    Le seuil ``min_score`` est strict : seuls les scores supérieurs sont retenus,
    ce qui exclut par défaut les termes non discriminants (score nul). ``k=None``
    conserve tous les termes dépassant le seuil ; ``k=0`` donne des listes vides.
    Les égalités sont départagées par ordre lexical pour un résultat reproductible.
    Le seuil est un score, pas une p-value ni un niveau de significativité.

    Pour un vocabulaire commun, prendre l'union des listes obtenues. Un même terme
    peut être sélectionné pour plusieurs classes. Les scores fournis ne sont pas
    modifiés. Exemple : ``select_discriminant_terms(compute_chi2_scores(docs), k=50)``.
    """
    if k is not None and (isinstance(k, bool) or not isinstance(k, int) or k < 0):
        raise ValueError("k doit être un entier positif ou nul, ou None")
    if not math.isfinite(min_score) or min_score < 0:
        raise ValueError("min_score doit être fini et positif ou nul")

    selected: dict[str, list[str]] = {}
    for label in sorted(scores):
        term_scores = scores[label]
        if any(not math.isfinite(score) or score < 0 for score in term_scores.values()):
            raise ValueError("Les scores Chi² doivent être finis et positifs ou nuls")
        ranked = sorted(
            (term for term, score in term_scores.items() if score > min_score),
            key=lambda term: (-term_scores[term], term),
        )
        selected[label] = ranked if k is None else ranked[:k]
    return selected

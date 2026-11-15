"""Index inversé des termes du corpus juridique arabe."""

from collections import Counter
from collections.abc import Iterable
from typing import TypeAlias

from src.common.document_schema import Document
from src.lexical_search.preprocessing import (
    DEFAULT_STOPWORDS,
    PreprocessingConfig,
    preprocess_text,
)

InvertedIndex: TypeAlias = dict[str, dict[str, int]]


def build_inverted_index(
    documents: Iterable[Document],
    config: PreprocessingConfig | None = None,
    *,
    stopwords: Iterable[str] = DEFAULT_STOPWORDS,
) -> InvertedIndex:
    """Construit la map ``terme -> {identifiant_document: fréquence}``.

    La fréquence est le nombre brut d'occurrences après prétraitement, pas une
    fréquence normalisée ni un poids TF-IDF. Les documents sans tokens ne créent
    aucune entrée. Le corpus et ses textes restent inchangés.

    Chaque document doit avoir un identifiant non vide et unique, y compris les
    documents vides ; les doublons lèvent ``ValueError`` pour éviter d'écraser
    silencieusement des fréquences. Les erreurs de prétraitement sont propagées.
    Les termes et identifiants sont triés pour obtenir un ordre reproductible,
    indépendamment de l'ordre des documents fournis.

    Utiliser les mêmes options et stopwords lors du prétraitement des requêtes.
    Exemple de résultat : ``{"عقد": {"doc-1": 2, "doc-2": 1}}``.
    """
    index: InvertedIndex = {}
    seen_ids: set[str] = set()
    # Matérialiser une seule fois pour accepter aussi un générateur de stopwords.
    selected_stopwords = tuple(stopwords)
    for document in documents:
        document_id = document["id"]
        if not isinstance(document_id, str):
            raise TypeError("L'identifiant du document doit être une chaîne")
        if not document_id.strip():
            raise ValueError("L'identifiant du document ne doit pas être vide")
        if document_id in seen_ids:
            raise ValueError(f"Identifiant de document dupliqué : {document_id}")
        seen_ids.add(document_id)

        frequencies = Counter(
            preprocess_text(document["text"], config, stopwords=selected_stopwords)
        )
        for term, frequency in frequencies.items():
            index.setdefault(term, {})[document_id] = frequency

    return {term: dict(sorted(postings.items())) for term, postings in sorted(index.items())}

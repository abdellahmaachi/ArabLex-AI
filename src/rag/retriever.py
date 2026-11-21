"""Récupération des Top-K passages par encodage de requête et recherche dense."""

import math
from typing import Any, Protocol

import numpy as np


class QueryEncoder(Protocol):
    """Interface satisfaite par AraBERTEncoder, sans importer le modèle ici."""

    dimension: int

    def encode_query(self, text: str) -> np.ndarray: ...


class PassageStore(Protocol):
    """Interface de recherche satisfaite par QdrantStore."""

    dimension: int

    def search(
        self, query_embedding: Any, *, top_k: int = 5,
        document_id: str | None = None, class_name: str | None = None,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]: ...


class RAGRetriever:
    """Relie un encodeur chargé à un stockage ouvert, sans modifier l'index.

    L'appelant gère le cycle de vie du modèle et du stockage. Leur dimension
    doit correspondre ; il doit également employer le même checkpoint et les
    mêmes conventions de préparation que lors de l'indexation. La dimension
    seule ne permet pas de vérifier cette compatibilité sémantique.

    Exemple ::

        retriever = RAGRetriever(encoder, store)
        hits = retriever.retrieve("ما هي حقوق الملكية؟", top_k=5)
        texts = [hit["passage"]["text"] for hit in hits]
    """

    def __init__(self, encoder: QueryEncoder, store: PassageStore) -> None:
        for name, component in (("encoder", encoder), ("store", store)):
            dimension = component.dimension
            if type(dimension) is not int or dimension < 1:
                raise ValueError(f"La dimension de {name} doit être un entier strictement positif")
        if encoder.dimension != store.dimension:
            raise ValueError("Dimensions incompatibles entre l'encodeur et le stockage")
        self.encoder = encoder
        self.store = store

    def retrieve(
        self, query: str, *, top_k: int = 5,
        document_id: str | None = None, class_name: str | None = None,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Renvoie au plus top_k résultats ``{id, score, passage}``.

        Les résultats suivent l'ordre cosinus décroissant fourni par Qdrant.
        Plusieurs passages d'un même document peuvent apparaître. Le payload
        complet est conservé, notamment texte, positions et métadonnées sources.
        Les filtres sont combinés par ET et le seuil de score est inclusif.
        Aucun résultat admissible (ou index vide) donne une liste vide.

        La requête non blanche est transmise telle quelle à l'encodeur : aucune
        normalisation supplémentaire ne risque de diverger du corpus. Les entrées
        invalides sont refusées avant l'inférence. Les erreurs du modèle ou du
        stockage sont propagées, et ne sont pas assimilées à une absence de résultats.
        """
        if not isinstance(query, str):
            raise TypeError("query doit être une chaîne")
        if not query.strip():
            raise ValueError("query doit être non vide")
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k doit être un entier strictement positif")
        for name, value in (("document_id", document_id), ("class_name", class_name)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} doit être une chaîne non vide")
        if score_threshold is not None:
            if isinstance(score_threshold, bool) or not isinstance(score_threshold, (int, float)):
                raise TypeError("score_threshold doit être un nombre réel")
            if not math.isfinite(score_threshold) or not -1 <= score_threshold <= 1:
                raise ValueError("score_threshold doit être compris entre -1 et 1")
        vector = self.encoder.encode_query(query)
        return self.store.search(
            vector, top_k=top_k, document_id=document_id,
            class_name=class_name, score_threshold=score_threshold,
        )

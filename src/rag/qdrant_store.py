"""Stockage local des passages RAG et recherche dense par similarité cosinus."""

import json
from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
from qdrant_client import QdrantClient, models


class QdrantStore:
    """Collection locale persistante, sans serveur ni appel réseau.

    ``path=None`` crée un stockage en mémoire. Sinon, le dossier est créé ou
    réouvert. Une collection existante doit avoir la dimension demandée et la
    distance cosinus ; elle n'est jamais recréée implicitement. Utiliser une
    collection distincte pour chaque modèle/version d'encodeur, même lorsque
    les dimensions sont identiques. Fermer le store pour libérer le verrou du
    dossier avant de le rouvrir (un seul client par dossier).

    Exemple ::

        with QdrantStore("artifacts/rag/qdrant", dimension=encoder.dimension) as store:
            store.upsert(passages, encoder.encode_documents(p["text"] for p in passages))
            hits = store.search(encoder.encode_query("حقوق الملكية"), top_k=5)
    """

    def __init__(
        self, path: str | Path | None = None, *, dimension: int,
        collection_name: str = "legal_passages",
        create_if_missing: bool = True,
    ) -> None:
        if type(dimension) is not int or dimension < 1:
            raise ValueError("dimension doit être un entier strictement positif")
        if not isinstance(collection_name, str) or not collection_name.strip():
            raise ValueError("collection_name doit être une chaîne non vide")
        self.dimension = dimension
        self.collection_name = collection_name
        self._closed = False
        if not create_if_missing and (path is None or not Path(path).is_dir()):
            raise FileNotFoundError("Stockage Qdrant existant requis")
        self._client = (
            QdrantClient(location=":memory:") if path is None
            else QdrantClient(path=str(Path(path).resolve()))
        )
        try:
            if self._client.collection_exists(collection_name):
                vectors = self._client.get_collection(collection_name).config.params.vectors
                if not isinstance(vectors, models.VectorParams) or (
                    vectors.size != dimension or vectors.distance != models.Distance.COSINE
                ):
                    raise ValueError("Collection incompatible : dimension ou distance différente")
            else:
                if not create_if_missing:
                    raise ValueError("Collection Qdrant introuvable")
                self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
                )
        except Exception:
            self.close()
            raise

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Le stockage Qdrant est fermé")

    def _vectors(self, values: Any, shape: tuple[int, ...]) -> np.ndarray:
        array = np.asarray(values)
        if array.shape != shape:
            raise ValueError(f"Dimensions attendues : {shape}, reçues : {array.shape}")
        if array.dtype.kind not in "fiu":
            raise TypeError("Les embeddings doivent contenir des nombres réels")
        array = array.astype(np.float64)
        if not np.isfinite(array).all():
            raise ValueError("Les embeddings doivent être finis")
        # Normalisation stable, même pour de très grandes/petites valeurs finies.
        scales = np.max(np.abs(array), axis=-1, keepdims=True)
        if (scales == 0).any():
            raise ValueError("Les embeddings nuls sont interdits")
        scaled = array / scales
        return (scaled / np.linalg.norm(scaled, axis=-1, keepdims=True)).astype(np.float32)

    def upsert(
        self, passages: Iterable[Mapping[str, Any]], embeddings: Any, *, batch_size: int = 128,
    ) -> int:
        """Insère/remplace les passages par UUID ; renvoie leur nombre.

        Les embeddings sont une matrice (nombre de passages, dimension). Les
        champs id, document_id et text sont requis ; tous les champs du passage
        sont conservés comme payload JSON. Le lot entier est validé avant toute
        écriture. Les écritures par lots ne forment pas une transaction : une
        panne peut laisser une partie des lots enregistrée ; relancer est idempotent.
        Les anciens passages absents de l'appel sont conservés. Pour réindexer
        un document modifié, utiliser explicitement ``delete_document`` auparavant.
        """
        self._ensure_open()
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size doit être un entier strictement positif")
        documents = list(passages)
        vectors = self._vectors(embeddings, (len(documents), self.dimension))
        points = []
        seen: set[str] = set()
        for passage in documents:
            if not isinstance(passage, Mapping):
                raise TypeError("Chaque passage doit être un objet")
            for key in ("id", "document_id", "text"):
                if not isinstance(passage.get(key), str) or not passage[key].strip():
                    raise ValueError(f"Champ {key} absent ou vide")
            point_id = str(UUID(passage["id"]))
            if point_id in seen:
                raise ValueError(f"Identifiant de passage dupliqué : {point_id}")
            seen.add(point_id)
            payload = json.loads(json.dumps(dict(passage), ensure_ascii=False, allow_nan=False))
            points.append(models.PointStruct(
                id=point_id, vector=vectors[len(points)].tolist(), payload=payload,
            ))
        for start in range(0, len(points), batch_size):
            self._client.upsert(
                collection_name=self.collection_name, points=points[start:start + batch_size], wait=True,
            )
        return len(points)

    def search(
        self, query_embedding: Any, *, top_k: int = 5,
        document_id: str | None = None, class_name: str | None = None,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Renvoie ``{id, score, passage}`` par score cosinus décroissant.

        Les filtres document_id et metadata.class sont combinés par ET. Le seuil
        est inclusif. Les vecteurs ne sont pas renvoyés ; le passage contient le
        texte, les positions et les métadonnées enregistrées lors de l'upsert.
        """
        self._ensure_open()
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k doit être un entier strictement positif")
        vector = self._vectors(query_embedding, (self.dimension,))
        if score_threshold is not None:
            if isinstance(score_threshold, bool) or not isinstance(score_threshold, (int, float)):
                raise TypeError("score_threshold doit être un nombre réel")
            if not np.isfinite(score_threshold) or not -1 <= score_threshold <= 1:
                raise ValueError("score_threshold doit être compris entre -1 et 1")
        conditions = []
        for key, value in (("document_id", document_id), ("metadata.class", class_name)):
            if value is not None:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"Le filtre {key} doit être une chaîne non vide")
                conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
        response = self._client.query_points(
            collection_name=self.collection_name, query=vector.tolist(), limit=top_k,
            query_filter=models.Filter(must=conditions) if conditions else None,
            score_threshold=score_threshold, with_payload=True, with_vectors=False,
        )
        return [
            {"id": str(point.id), "score": float(point.score), "passage": deepcopy(point.payload)}
            for point in response.points
        ]

    def delete_document(self, document_id: str) -> None:
        """Supprime tous les passages d'un document, sans toucher aux autres."""
        self._ensure_open()
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("document_id doit être une chaîne non vide")
        self._client.delete(
            collection_name=self.collection_name, wait=True,
            points_selector=models.FilterSelector(filter=models.Filter(must=[
                models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)),
            ])),
        )

    def count(self) -> int:
        """Nombre exact de passages enregistrés."""
        self._ensure_open()
        return self._client.count(collection_name=self.collection_name, exact=True).count

    def close(self) -> None:
        """Libère les ressources et le verrou local ; plusieurs appels sont permis."""
        if not self._closed:
            self._client.close()
            self._closed = True

    def __enter__(self) -> "QdrantStore":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

"""Recherche exacte dans les annotations publiées par build_ner_index.py."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.ner.corpus_indexer import build_entity_index


class NERService:
    """Index en mémoire, sans modèle AraBERT ni réannotation du corpus."""

    def __init__(self, documents: list[dict[str, Any]]) -> None:
        # Valide identifiants, annotations et positions dans le texte original.
        self._index = build_entity_index(documents)
        required = ("class", "filename", "path")
        for document in documents:
            if any(not isinstance(document.get(key), str) for key in required):
                raise ValueError("Métadonnées du document NER incomplètes")
            if not document["class"].strip():
                raise ValueError("Classe du document NER vide")
        self._documents = {document["id"]: deepcopy(document) for document in documents}

    @classmethod
    def load(cls, path: str | Path) -> "NERService":
        """Charge et valide le format exporté, y compris l'index inversé."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or (
            type(payload.get("schema_version")) is not int
            or payload["schema_version"] != 1
            or payload.get("offset_unit") != "python_character"
            or payload.get("end_exclusive") is not True
            or payload.get("entity_key") != "exact_text"
            or not isinstance(payload.get("documents"), list)
        ):
            raise ValueError("Format d'index NER non reconnu")
        service = cls(payload["documents"])
        if payload.get("entity_index") != service._index:
            raise ValueError("Index NER incohérent avec les annotations")
        return service

    def search(
        self, text: str, *, label: str | None = None,
        class_name: str | None = None, k: int = 10,
    ) -> dict[str, Any]:
        """Renvoie les documents et uniquement leurs occurrences correspondantes.

        Correspondance exacte (espaces et diacritiques significatifs), filtres
        combinés par ET. Classement par nombre d'occurrences décroissant, puis
        identifiant du document. Un filtre inconnu donne une liste vide.
        """
        for name, value in (("text", text), ("label", label), ("class_name", class_name)):
            if value is None and name != "text":
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} doit être une chaîne non vide")
        if type(k) is not int or k < 1:
            raise ValueError("k doit être un entier strictement positif")
        matches: dict[str, list[dict[str, Any]]] = {}
        labels = [label] if label is not None else self._index
        for entity_label in labels:
            postings = self._index.get(entity_label, {}).get(text, {})
            for doc_id, positions in postings.items():
                if class_name is not None and self._documents[doc_id]["class"] != class_name:
                    continue
                matches.setdefault(doc_id, []).extend(
                    {"text": text, "label": entity_label, **position} for position in positions
                )
        ranked = sorted(matches, key=lambda doc_id: (-len(matches[doc_id]), doc_id))[:k]
        results = []
        for doc_id in ranked:
            document = self._documents[doc_id]
            results.append({
                **{key: document[key] for key in ("id", "class", "filename", "path", "text")},
                "entities": sorted(matches[doc_id], key=lambda entity: entity["start"]),
            })
        return {"text": text, "results": results}

"""Annotation du corpus et export d'un index inversé des entités NER."""

import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from copy import deepcopy
from numbers import Integral
from pathlib import Path
from typing import Any, Protocol, TypeAlias, TypedDict

from src.ner.bio_decoder import Entity, validate_bio_labels


class Annotator(Protocol):
    """Interface satisfaite par AraBERTAnnotator, sans charger PyTorch ici."""

    def annotate(self, text: str) -> list[Entity]: ...


class Position(TypedDict):
    start: int
    end: int


EntityIndex: TypeAlias = dict[str, dict[str, dict[str, list[Position]]]]


def _documents(documents: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Valide les identifiants avant toute inférence et copie les métadonnées."""
    result = []
    seen = set()
    for document in documents:
        if not isinstance(document, Mapping):
            raise TypeError("Chaque document doit être un objet avec id et text")
        document_id = document.get("id")
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("Chaque document doit avoir un id textuel non vide")
        if document_id in seen:
            raise ValueError(f"Identifiant de document dupliqué : {document_id}")
        if not isinstance(document.get("text"), str):
            raise TypeError(f"{document_id} : text doit être une chaîne")
        seen.add(document_id)
        result.append(deepcopy(dict(document)))
    return sorted(result, key=lambda document: document["id"])


def _entities(text: str, entities: Any, document_id: str) -> list[Entity]:
    if not isinstance(entities, list):
        raise TypeError(f"{document_id} : entities doit être une liste")
    result: list[Entity] = []
    for entity in entities:
        if not isinstance(entity, Mapping) or not all(
            key in entity for key in ("text", "label", "start", "end")
        ):
            raise ValueError(f"{document_id} : entité incomplète")
        start, end, label = entity["start"], entity["end"], entity["label"]
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in (start, end)):
            raise ValueError(f"{document_id} : positions non entières")
        if not 0 <= start < end <= len(text):
            raise ValueError(f"{document_id} : positions hors limites")
        if not isinstance(label, str) or not label:
            raise ValueError(f"{document_id} : type d'entité vide ou invalide")
        validate_bio_labels([f"B-{label}"])
        if entity["text"] != text[start:end]:
            raise ValueError(f"{document_id} : texte d'entité différent de text[start:end]")
        result.append(Entity(text=entity["text"], label=label, start=int(start), end=int(end)))
    result.sort(key=lambda entity: (entity["start"], entity["end"], entity["label"]))
    previous_end = 0
    for entity in result:
        if entity["start"] < previous_end:
            raise ValueError(f"{document_id} : entités dupliquées ou chevauchantes")
        previous_end = entity["end"]
    return result


def annotate_documents(
    documents: Iterable[Mapping[str, Any]], annotator: Annotator,
) -> list[dict[str, Any]]:
    """Annote chaque document, une fois, avec un annotateur déjà chargé.

    Accepte notamment la sortie de ``load_documents``. Les champs id et text
    sont obligatoires ; toutes les métadonnées sont copiées. Le champ entities
    est remplacé par les nouvelles annotations. Les entrées restent inchangées.
    Un échec interrompt le traitement, sans ignorer silencieusement un document.
    Les documents et entités sont triés pour un export reproductible.
    """
    annotated = _documents(documents)
    for document in annotated:
        document["entities"] = _entities(
            document["text"], annotator.annotate(document["text"]), document["id"],
        )
    return annotated


def _validated_annotations(documents: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    annotated = _documents(documents)
    for document in annotated:
        document["entities"] = _entities(
            document["text"], document.get("entities"), document["id"],
        )
    return annotated


def _index(annotated: list[dict[str, Any]]) -> EntityIndex:
    index: EntityIndex = {}
    for document in annotated:
        for entity in document["entities"]:
            postings = index.setdefault(entity["label"], {}).setdefault(entity["text"], {})
            postings.setdefault(document["id"], []).append(
                Position(start=entity["start"], end=entity["end"])
            )
    return {
        label: {text: dict(sorted(postings.items())) for text, postings in sorted(terms.items())}
        for label, terms in sorted(index.items())
    }


def build_entity_index(annotated_documents: Iterable[Mapping[str, Any]]) -> EntityIndex:
    """Construit ``type -> texte exact -> id_document -> [{start, end}, ...]``.

    Aucune normalisation : diacritiques et espaces sont significatifs. Un même
    texte peut apparaître sous plusieurs types. Chaque occurrence est conservée
    avec sa position Python, fin exclusive ; la fréquence est len(positions).
    Les documents sans entités ne créent aucune entrée dans l'index.
    """
    return _index(_validated_annotations(annotated_documents))


def _bundle(annotated: list[dict[str, Any]]) -> dict[str, Any]:
    index = _index(annotated)
    counts = Counter(entity["label"] for document in annotated for entity in document["entities"])
    return {
        "schema_version": 1,
        "offset_unit": "python_character",
        "end_exclusive": True,
        "entity_key": "exact_text",
        "documents": annotated,
        "entity_index": index,
        "statistics": {
            "documents": len(annotated),
            "documents_with_entities": sum(bool(document["entities"]) for document in annotated),
            "entities": sum(counts.values()),
            "unique_entities": sum(len(terms) for terms in index.values()),
            "entities_by_type": dict(sorted(counts.items())),
        },
    }


def _write_json(bundle: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    # Refuser explicitement un chemin qui écraserait un fichier du corpus.
    target = path.resolve()
    for document in bundle["documents"]:
        if document.get("path") and Path(document["path"]).resolve() == target:
            raise ValueError("Le fichier de sortie ne doit pas remplacer un document source")
    # Sérialiser avant de créer quoi que ce soit : métadonnées non JSON refusées.
    payload = json.dumps(bundle, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def export_annotations(
    annotated_documents: Iterable[Mapping[str, Any]], output_path: str | Path,
) -> Path:
    """Exporte annotations, index et statistiques en un fichier JSON UTF-8.

    Les annotations sont validées et l'index recalculé avant l'écriture. Un
    fichier existant est remplacé atomiquement, jamais complété par ajout.
    Les métadonnées des documents doivent être sérialisables en JSON.
    """
    return _write_json(_bundle(_validated_annotations(annotated_documents)), output_path)


def index_corpus(
    documents: Iterable[Mapping[str, Any]],
    annotator: Annotator,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Annote, indexe et éventuellement exporte le corpus en une seule opération.

    Exemple ::

        annotator = AraBERTAnnotator("artifacts/ner/mon_run/best_model")
        result = index_corpus(load_documents(strict=True), annotator,
                              output_path="artifacts/ner/corpus_entities.json")

    Le modèle est réutilisé sur tout le corpus. Les documents vides ou sans
    entités sont conservés dans documents. Aucun modèle ni source n'est modifié.
    """
    bundle = _bundle(annotate_documents(documents, annotator))
    if output_path is not None:
        _write_json(bundle, output_path)
    return bundle

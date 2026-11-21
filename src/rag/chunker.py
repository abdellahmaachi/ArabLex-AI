"""Segmentation déterministe des documents RAG en fenêtres de mots."""

import json
import re
from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any, TypedDict
from uuid import NAMESPACE_URL, uuid5

_WORDS = re.compile(r"\S+")


class Passage(TypedDict):
    """Positions Python, fin exclusive, dans le champ indiqué par text_field."""

    id: str
    document_id: str
    chunk_index: int
    text: str
    text_field: str
    start: int
    end: int
    word_count: int
    metadata: dict[str, Any]


def _validate_options(chunk_size: int, overlap: int) -> None:
    for name, value in (("chunk_size", chunk_size), ("overlap", overlap)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} doit être un entier")
    if chunk_size < 1:
        raise ValueError("chunk_size doit être strictement positif")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap doit vérifier 0 <= overlap < chunk_size")


def _document_id(document: Mapping[str, Any]) -> str:
    if not isinstance(document, Mapping):
        raise TypeError("Chaque document doit être un objet avec id et text")
    document_id = document.get("id")
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("Chaque document doit avoir un id textuel non vide")
    return document_id


def chunk_document(
    document: Mapping[str, Any], *, chunk_size: int = 200, overlap: int = 40,
) -> list[Passage]:
    """Découpe un document sans mutation, normalisation ni perte de mots.

    La taille et le chevauchement comptent les suites de caractères non blancs,
    pas les sous-tokens AraBERT. Les mots ne sont jamais coupés. Les espaces et
    paragraphes à l'intérieur d'un passage sont conservés exactement.

    Utilise ``prepared_text`` s'il existe, sinon ``text``. Un champ présent mais
    invalide est refusé ; un texte vide/blanc produit une liste vide. ``start`` et
    ``end`` permettent de retrouver le passage par ``document[text_field][start:end]``.
    Les positions du texte préparé ne sont pas celles du texte original.

    ``chunk_index`` commence à zéro. L'UUID5 dépend du document, du champ, des
    positions et du contenu du passage. Les autres champs du document (classe,
    fichier, chemin et métadonnées supplémentaires) sont copiés dans ``metadata``.
    Chaque passage possède sa propre copie profonde de ces métadonnées.
    """
    _validate_options(chunk_size, overlap)
    document_id = _document_id(document)
    text_field = "prepared_text" if "prepared_text" in document else "text"
    text = document.get(text_field)
    if not isinstance(text, str):
        raise TypeError(f"{document_id} : {text_field} doit être une chaîne")
    words = list(_WORDS.finditer(text))
    metadata = {
        key: value for key, value in document.items()
        if key not in {"id", "text", "prepared_text"}
    }
    passages: list[Passage] = []
    for first in range(0, len(words), chunk_size - overlap):
        last = min(first + chunk_size, len(words))
        start, end = words[first].start(), words[last - 1].end()
        content = text[start:end]
        identity = json.dumps(
            ["arabic-legal-text-mining/passage", document_id, text_field, start, end, content],
            ensure_ascii=False, separators=(",", ":"),
        )
        passages.append(Passage(
            id=str(uuid5(NAMESPACE_URL, identity)),
            document_id=document_id,
            chunk_index=len(passages),
            text=content,
            text_field=text_field,
            start=start,
            end=end,
            word_count=last - first,
            metadata=deepcopy(metadata),
        ))
        # Ne pas créer un dernier passage entièrement contenu dans le précédent.
        if last == len(words):
            break
    return passages


def chunk_documents(
    documents: Iterable[Mapping[str, Any]], *, chunk_size: int = 200, overlap: int = 40,
) -> list[Passage]:
    """Segmente le corpus dans l'ordre d'entrée, sans mélanger les documents.

    Les identifiants de documents dupliqués sont refusés, même pour un texte vide.
    Exemple : ``chunk_documents(load_rag_documents(), chunk_size=200, overlap=40)``.
    """
    _validate_options(chunk_size, overlap)
    passages: list[Passage] = []
    seen: set[str] = set()
    for document in documents:
        document_id = _document_id(document)
        if document_id in seen:
            raise ValueError(f"Identifiant de document dupliqué : {document_id}")
        seen.add(document_id)
        passages.extend(chunk_document(document, chunk_size=chunk_size, overlap=overlap))
    return passages

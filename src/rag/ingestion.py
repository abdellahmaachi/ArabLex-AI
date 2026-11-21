"""Chargement et préparation du corpus RAG, sans modèle ni accès réseau.

``text`` reste la source des citations ; ``prepared_text`` est destiné au
découpage puis à l'encodage. Ses positions peuvent différer du texte original.
Aucun filtrage lexical ni stemming n'est appliqué.
"""

import logging
from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.common.arabic_normalizer import NormalizationConfig, normalize_arabic
from src.common.document_loader import DEFAULT_DATA_DIR, load_documents

LOGGER = logging.getLogger(__name__)
DEFAULT_NORMALIZATION = NormalizationConfig(
    remove_diacritics=False, remove_tatweel=False, normalize_alef=False,
)


def prepare_documents(
    documents: Iterable[Mapping[str, Any]],
    *,
    normalization: NormalizationConfig | None = None,
) -> list[dict[str, Any]]:
    """Copie et prépare les documents dans leur ordre d'entrée.

    Chaque document doit avoir un ``id`` textuel non vide et unique et un
    ``text`` textuel. Les métadonnées sont copiées en profondeur et le champ
    ``prepared_text`` est recalculé sans modifier le texte source.

    Par défaut : NFC, suppression du BOM initial et des espaces aux extrémités,
    uniformisation CRLF/CR en LF. Lettres, diacritiques, chiffres, ponctuation
    et paragraphes internes sont conservés. Une configuration explicite permet
    une normalisation arabe supplémentaire. Les textes vides après préparation
    sont ignorés avec un avertissement. Deux sources identiques restent distinctes.
    """
    if normalization is not None and not isinstance(normalization, NormalizationConfig):
        raise TypeError("normalization doit être une NormalizationConfig")
    config = DEFAULT_NORMALIZATION if normalization is None else normalization
    prepared: list[dict[str, Any]] = []
    seen: set[str] = set()
    for document in documents:
        if not isinstance(document, Mapping):
            raise TypeError("Chaque document doit être un objet avec id et text")
        document_id = document.get("id")
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("Chaque document doit avoir un id textuel non vide")
        if document_id in seen:
            raise ValueError(f"Identifiant de document dupliqué : {document_id}")
        text = document.get("text")
        if not isinstance(text, str):
            raise TypeError(f"{document_id} : text doit être une chaîne")
        seen.add(document_id)
        cleaned = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
        cleaned = normalize_arabic(cleaned, config).strip()
        if not cleaned:
            LOGGER.warning("Document vide ignoré : %s", document_id)
            continue
        result = deepcopy(dict(document))
        result["prepared_text"] = cleaned
        prepared.append(result)
    return prepared


def load_rag_documents(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    strict: bool = True,
    normalization: NormalizationConfig | None = None,
) -> list[dict[str, Any]]:
    """Charge les fichiers .txt UTF-8 du corpus et prépare leur texte pour le RAG.

    Réutilise les classes, métadonnées, UUID stables et l'ordre du chargeur commun.
    Par défaut, une erreur de lecture interrompt l'ingestion pour éviter un index
    incomplet. Avec ``strict=False``, les fichiers illisibles sont signalés puis
    ignorés. Un dossier de classe absent provoque toujours une exception.

    Exemple : ``documents = load_rag_documents("data/raw")``.
    Le découpage, l'encodage et l'indexation sont des étapes ultérieures.
    """
    return prepare_documents(load_documents(data_dir, strict=strict), normalization=normalization)

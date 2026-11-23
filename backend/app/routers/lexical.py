"""Recherche lexicale sur les artefacts publiés par build_lexical_index.py."""

import json
import logging
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from backend.app.core.config import Settings, get_settings
from backend.app.schemas.lexical import LexicalSearchRequest, LexicalSearchResponse
from src.lexical_search.service import LexicalSearchService

router = APIRouter(prefix="/api/search", tags=["lexical"])
LOGGER = logging.getLogger(__name__)
_load_lock = Lock()


@lru_cache(maxsize=1)
def _load_service(path: Path) -> LexicalSearchService:
    # Les artefacts locaux configurés doivent être de confiance (format pickle).
    return LexicalSearchService.load(path)


def get_lexical_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> LexicalSearchService:
    """Relit le pointeur et ne charge qu'une fois chaque construction active.

    Le verrou évite les doubles chargements lors de requêtes concurrentes.
    Une nouvelle publication current.json devient visible à la requête suivante.
    Aucun entraînement ni écriture n'est effectué depuis l'API.
    """
    try:
        pointer_path = settings.lexical_index_path
        if pointer_path is None:
            raise ValueError("Chemin de l'index lexical absent")
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        if not isinstance(pointer, dict) or type(pointer.get("version")) is not int:
            raise ValueError("Pointeur lexical invalide")
        directory = pointer.get("directory")
        if pointer["version"] != 1 or not isinstance(directory, str) or not directory.strip():
            raise ValueError("Pointeur lexical invalide")
        root = pointer_path.parent.resolve()
        build = (root / directory).resolve()
        artifact = (build / "service.joblib").resolve()
        if build == root or not build.is_relative_to(root) or not artifact.is_relative_to(build):
            raise ValueError("Le pointeur doit désigner une construction dans le dossier d'index")
        with _load_lock:
            return _load_service(artifact)
    except Exception as exc:
        # Inclut les erreurs de désérialisation et les incompatibilités d'artefacts.
        LOGGER.exception("Impossible de charger l'index lexical")
        raise HTTPException(status_code=503, detail="Index lexical indisponible") from exc


@router.post("/lexical", response_model=LexicalSearchResponse)
def search_lexical(
    request: LexicalSearchRequest,
    service: Annotated[LexicalSearchService, Depends(get_lexical_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> LexicalSearchResponse:
    """Oriente par Chi² et classe les documents par similarité TF-IDF/cosinus."""
    k = request.k if "k" in request.model_fields_set else settings.lexical_top_k
    try:
        result = service.search(
            request.query, k=k, min_score=request.min_score, class_name=request.class_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return LexicalSearchResponse.model_validate(result)

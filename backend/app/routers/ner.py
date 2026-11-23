"""Recherche par entité dans l'index NER local."""

import logging
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from backend.app.core.config import Settings, get_settings
from backend.app.schemas.ner import EntitySearchRequest, EntitySearchResponse
from src.ner.service import NERService

router = APIRouter(prefix="/api/ner", tags=["ner"])
LOGGER = logging.getLogger(__name__)
_load_lock = Lock()


@lru_cache(maxsize=1)
def _load_service(path: Path, modified_ns: int, size: int) -> NERService:
    # Date et taille invalident le cache après remplacement de l'export JSON.
    return NERService.load(path)


def get_ner_service(settings: Annotated[Settings, Depends(get_settings)]) -> NERService:
    """Charge l'index à la demande, une seule fois par version et processus."""
    try:
        if settings.ner_index_path is None:
            raise ValueError("NER_INDEX_PATH doit être configuré")
        path = settings.ner_index_path.resolve()
        with _load_lock:
            stat = path.stat()
            return _load_service(path, stat.st_mtime_ns, stat.st_size)
    except Exception as exc:
        LOGGER.exception("Impossible de charger l'index NER")
        raise HTTPException(status_code=503, detail="Index NER indisponible") from exc


@router.post("/search", response_model=EntitySearchResponse)
def search_ner(
    request: EntitySearchRequest,
    service: Annotated[NERService, Depends(get_ner_service)],
) -> EntitySearchResponse:
    """Recherche exacte avec filtres facultatifs par type d'entité et classe."""
    result = service.search(
        request.text, label=request.label, class_name=request.class_name, k=request.k,
    )
    return EntitySearchResponse.model_validate(result)

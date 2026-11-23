"""Questions RAG sur une construction locale publiée, avec génération Gemini."""

import json
import logging
from contextlib import ExitStack
from pathlib import Path
from threading import Lock
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from backend.app.core.config import Settings, get_settings
from backend.app.schemas.rag import RAGRequest, RAGResponse
from src.common.arabic_normalizer import NormalizationConfig
from src.rag.arabert_encoder import AraBERTEncoder
from src.rag.gemini_client import GeminiClient, GeminiError, GeminiRequestError
from src.rag.ingestion import prepare_documents
from src.rag.qdrant_store import QdrantStore
from src.rag.retriever import RAGRetriever
from src.rag.service import InvalidCitationError, RAGService

router = APIRouter(prefix="/api/rag", tags=["rag"])
LOGGER = logging.getLogger(__name__)
_lock = Lock()
_service: RAGService | None = None
_resources: ExitStack | None = None


class _QueryEncoder:
    """Applique à la requête la préparation utilisée pendant l'indexation."""

    def __init__(self, encoder: AraBERTEncoder, normalization: NormalizationConfig):
        self.encoder = encoder
        self.normalization = normalization
        self.dimension = encoder.dimension

    def encode_query(self, text: str):
        prepared = prepare_documents(
            [{"id": "query", "text": text}], normalization=self.normalization,
        )
        if not prepared:
            raise ValueError("Question vide après préparation")
        return self.encoder.encode_query(prepared[0]["prepared_text"])


def _build_service(settings: Settings, resources: ExitStack) -> RAGService:
    if settings.gemini_api_key is None or not settings.gemini_model:
        raise ValueError("Configuration Gemini absente")
    if settings.rag_index_path is None:
        raise ValueError("RAG_INDEX_PATH doit être configuré")
    pointer_path = settings.rag_index_path
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    if not isinstance(pointer, dict) or pointer.get("schema_version") != 1:
        raise ValueError("Pointeur RAG invalide")
    root = pointer_path.parent.resolve()
    build = (root / pointer["build_dir"]).resolve()
    manifest_path = (root / pointer["manifest"]).resolve()
    store_path = (root / pointer["qdrant_path"]).resolve()
    if (build == root or not build.is_relative_to(root)
            or not manifest_path.is_relative_to(build) or not store_path.is_relative_to(build)):
        raise ValueError("Construction RAG hors du dossier d'index")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1 or manifest.get("distance") != "Cosine"
            or manifest["dimension"] != pointer["dimension"]
            or manifest["collection_name"] != pointer["collection_name"]
            or Path(manifest["qdrant_path"]).resolve() != store_path):
        raise ValueError("Manifeste RAG incompatible avec le pointeur")
    normalization = NormalizationConfig(**manifest["normalization"])
    encoder = AraBERTEncoder(
        manifest["model_dir"], device=settings.inference_device,
        batch_size=settings.rag_batch_size, max_length=manifest["max_length"],
        truncate=manifest["truncate"],
    )
    if encoder.dimension != manifest["dimension"]:
        raise ValueError("Dimension de l'encodeur incompatible avec l'index")
    store = resources.enter_context(QdrantStore(
        store_path, dimension=encoder.dimension,
        collection_name=manifest["collection_name"], create_if_missing=False,
    ))
    generator = resources.enter_context(GeminiClient(
        api_key=settings.gemini_api_key.get_secret_value(), model=settings.gemini_model,
        timeout_ms=settings.gemini_timeout_ms, max_attempts=settings.gemini_max_attempts,
        max_output_tokens=settings.gemini_max_output_tokens,
        temperature=settings.gemini_temperature,
    ))
    return RAGService(RAGRetriever(_QueryEncoder(encoder, normalization), store), generator)


def get_rag_service(settings: Annotated[Settings, Depends(get_settings)]) -> RAGService:
    """Initialisation différée ; redémarrer l'API pour changer d'index/modèle.

    Qdrant Local exige un seul processus API par dossier de stockage.
    """
    global _service, _resources
    with _lock:
        if _service is None:
            try:
                with ExitStack() as resources:
                    service = _build_service(settings, resources)
                    _resources = resources.pop_all()
                    _service = service
            except Exception as exc:
                # Ne pas journaliser les exceptions SDK : elles peuvent contenir des secrets.
                LOGGER.error("Initialisation RAG impossible (%s)", type(exc).__name__)
                raise HTTPException(status_code=503, detail="Service RAG indisponible") from None
        return _service


def close_rag_service() -> None:
    """Libère les clients Gemini et Qdrant, même après une initialisation partielle."""
    global _service, _resources
    with _lock:
        resources, _resources = _resources, None
        _service = None
        if resources is not None:
            resources.close()


@router.post("/ask", response_model=RAGResponse)
def ask_rag(
    request: RAGRequest,
    service: Annotated[RAGService, Depends(get_rag_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> RAGResponse:
    """Renvoie la réponse, les passages sources et le contrôle des citations."""
    top_k = request.top_k if "top_k" in request.model_fields_set else settings.rag_top_k
    try:
        # Sérialiser l'accès au modèle et au stockage local partagés.
        with _lock:
            result = service.answer(
                request.question, top_k=top_k, document_id=request.document_id,
                class_name=request.class_name, score_threshold=request.score_threshold,
            )
    except GeminiRequestError as exc:
        raise HTTPException(
            status_code=503 if exc.retryable else 502, detail="Service de génération indisponible",
        ) from None
    except (GeminiError, InvalidCitationError):
        raise HTTPException(status_code=502, detail="Réponse de génération invalide") from None
    except ValueError:
        raise HTTPException(status_code=422, detail="Question incompatible avec le service RAG") from None
    return RAGResponse.model_validate(result)

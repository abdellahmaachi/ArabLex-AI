"""Classification de textes avec un prédicteur Naive Bayes exporté."""

import logging
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from backend.app.core.config import Settings, get_settings
from backend.app.schemas.classification import ClassificationRequest, ClassificationResponse
from src.classification.predictor import NaiveBayesPredictor

router = APIRouter(prefix="/api", tags=["classification"])
LOGGER = logging.getLogger(__name__)
_load_lock = Lock()


@lru_cache(maxsize=1)
def _load_predictor(path: Path) -> NaiveBayesPredictor:
    # Charger uniquement les exports locaux de confiance : joblib utilise pickle.
    return NaiveBayesPredictor.load(path)


def get_classification_predictor(
    settings: Annotated[Settings, Depends(get_settings)],
) -> NaiveBayesPredictor:
    """Charge CLASSIFICATION_MODEL_PATH à la première requête, sans entraînement.

    Le cache est partagé entre les requêtes du processus et protégé contre les
    doubles chargements concurrents. Pour remplacer un modèle, configurer un
    nouveau chemin ou redémarrer le processus si le fichier est écrasé en place.
    """
    try:
        path = settings.classification_model_path
        if path is None:
            raise ValueError("CLASSIFICATION_MODEL_PATH doit être configuré")
        with _load_lock:
            return _load_predictor(path.resolve())
    except Exception as exc:
        # Les erreurs de lecture/désérialisation ne sont pas des erreurs client.
        LOGGER.exception("Impossible de charger le prédicteur Naive Bayes")
        raise HTTPException(
            status_code=503, detail="Modèle de classification indisponible",
        ) from exc


@router.post("/classify", response_model=ClassificationResponse)
def classify(
    request: ClassificationRequest,
    predictor: Annotated[NaiveBayesPredictor, Depends(get_classification_predictor)],
) -> ClassificationResponse:
    """Renvoie la classe prédite et les probabilités de toutes les classes."""
    return ClassificationResponse(
        label=predictor.predict(request.text),
        probabilities=predictor.predict_proba(request.text),
    )

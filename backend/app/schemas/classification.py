"""Classification d'un texte par le prédicteur Naive Bayes."""

from .common import APIModel, NonBlankText, Probability


class ClassificationRequest(APIModel):
    text: NonBlankText


class ClassificationResponse(APIModel):
    """Classe prédite et probabilités renvoyées par predict_proba.

    Les clés entières des classes deviennent des chaînes dans le JSON.
    """

    label: str | int
    probabilities: dict[str | int, Probability]

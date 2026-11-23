"""Annotation NER et contrat de la future recherche dans l'index d'entités."""

from typing import Self

from pydantic import model_validator

from .common import APIModel, Document, NonBlankText, NonNegativeInt, PositiveInt


class NERRequest(APIModel):
    text: NonBlankText


class Entity(APIModel):
    """Indices en caractères Python dans le texte original, fin exclusive."""

    text: NonBlankText
    label: NonBlankText
    start: NonNegativeInt
    end: PositiveInt

    @model_validator(mode="after")
    def validate_span(self) -> Self:
        if self.end <= self.start:
            raise ValueError("end doit être strictement supérieur à start")
        return self


class NERResponse(APIModel):
    text: str
    entities: list[Entity]

    @model_validator(mode="after")
    def validate_entities(self) -> Self:
        previous_end = 0
        for entity in self.entities:
            if entity.start < previous_end or entity.end > len(self.text):
                raise ValueError("Entités non ordonnées, chevauchantes ou hors du texte")
            if self.text[entity.start:entity.end] != entity.text:
                raise ValueError("Le texte de l'entité ne correspond pas à ses positions")
            previous_end = entity.end
        return self


class EntitySearchRequest(APIModel):
    """Recherche d'une entité textuelle, éventuellement filtrée par type/classe."""

    text: NonBlankText
    label: NonBlankText | None = None
    class_name: NonBlankText | None = None
    k: PositiveInt = 10


class EntitySearchHit(Document):
    entities: list[Entity]


class EntitySearchResponse(APIModel):
    text: str
    results: list[EntitySearchHit]

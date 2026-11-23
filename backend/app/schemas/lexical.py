"""Contrats alignés sur LexicalSearchService.search."""

from typing import Annotated

from pydantic import Field

from .common import APIModel, CosineScore, Document, NonBlankText, NonNegativeInt


class LexicalSearchRequest(APIModel):
    query: NonBlankText
    k: NonNegativeInt = 10
    min_score: CosineScore = 0.0
    class_name: NonBlankText | None = None


class LexicalHit(Document):
    score: CosineScore


class LexicalSearchResponse(APIModel):
    query: str
    tokens: list[str]
    selected_class: str | None
    class_scores: dict[str, Annotated[float, Field(ge=0)]]
    results: list[LexicalHit]

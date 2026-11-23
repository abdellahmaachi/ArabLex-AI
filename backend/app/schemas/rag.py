"""Contrats alignés sur RAGService.answer et les passages du chunker."""

from typing import Any, Literal, Self

from pydantic import Field, model_validator

from .common import APIModel, CosineScore, NonBlankText, NonNegativeInt, PositiveInt


class RAGRequest(APIModel):
    question: NonBlankText
    top_k: PositiveInt = 5
    document_id: NonBlankText | None = None
    class_name: NonBlankText | None = None
    score_threshold: CosineScore | None = None


class Passage(APIModel):
    id: NonBlankText
    document_id: NonBlankText
    chunk_index: NonNegativeInt
    text: NonBlankText
    text_field: NonBlankText
    start: NonNegativeInt
    end: PositiveInt
    word_count: PositiveInt
    metadata: dict[str, Any]

    @model_validator(mode="after")
    def validate_span(self) -> Self:
        if self.end <= self.start:
            raise ValueError("end doit être strictement supérieur à start")
        return self


class RAGSource(APIModel):
    reference: str = Field(pattern=r"^S[1-9][0-9]*$")
    id: NonBlankText
    score: CosineScore
    passage: Passage
    cited: bool


class GenerationInfo(APIModel):
    model: NonBlankText
    finish_reason: str
    prompt_tokens: NonNegativeInt | None = None
    output_tokens: NonNegativeInt | None = None
    total_tokens: NonNegativeInt | None = None


class RAGResponse(APIModel):
    question: str
    answer: NonBlankText
    sources: list[RAGSource]
    cited_references: list[str]
    citation_status: Literal["valid", "missing", "not_applicable"]
    generation: GenerationInfo | None

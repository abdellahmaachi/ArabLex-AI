"""Validation des contrats HTTP sans charger les services NLP."""

import pytest
from pydantic import ValidationError

from backend.app.schemas import (
    ClassificationRequest,
    ClassificationResponse,
    LexicalSearchRequest,
    LexicalSearchResponse,
    NERRequest,
    NERResponse,
    RAGRequest,
    RAGResponse,
)


@pytest.mark.parametrize("model,field", [
    (ClassificationRequest, "text"), (LexicalSearchRequest, "query"),
    (NERRequest, "text"), (RAGRequest, "question"),
])
def test_requests_preserve_text_and_reject_blank_or_unknown_fields(model, field):
    original = "  نص عربي  "
    assert getattr(model(**{field: original}), field) == original
    for value in ("", " \n\t", None, 123):
        with pytest.raises(ValidationError):
            model(**{field: value})
    with pytest.raises(ValidationError):
        model(**{field: original, "unknown": True})


@pytest.mark.parametrize("options", [
    {"top_k": 0}, {"top_k": True}, {"top_k": 1.5},
    {"score_threshold": float("nan")}, {"score_threshold": 1.1},
    {"document_id": " "},
])
def test_invalid_rag_options(options):
    with pytest.raises(ValidationError):
        RAGRequest(question="سؤال", **options)


def test_lexical_contract_and_class_alias():
    assert LexicalSearchRequest(query="عقد", k=0).k == 0
    response = LexicalSearchResponse.model_validate({
        "query": "عقد", "tokens": ["عقد"], "selected_class": "civil",
        "class_scores": {"civil": 4.2},
        "results": [{"id": "1", "class": "civil", "filename": "1.txt",
                     "path": "civil/1.txt", "text": "عقد", "score": 0.8}],
    })
    assert response.results[0].class_name == "civil"
    assert response.model_dump(by_alias=True)["results"][0]["class"] == "civil"


def test_classification_probabilities():
    response = ClassificationResponse(label="civil", probabilities={"civil": 0.8, "penal": 0.2})
    assert ClassificationResponse.model_validate_json(response.model_dump_json()) == response
    with pytest.raises(ValidationError):
        ClassificationResponse(label="civil", probabilities={"civil": 1.1})


def test_ner_offsets_reference_original_text():
    payload = {"text": "  محمد", "entities": [
        {"text": "محمد", "label": "PER", "start": 2, "end": 6},
    ]}
    assert NERResponse.model_validate(payload).entities[0].start == 2
    payload["entities"][0]["end"] = 7
    with pytest.raises(ValidationError):
        NERResponse.model_validate(payload)


def test_rag_response_with_and_without_sources():
    response = RAGResponse(
        question="سؤال", answer="لا توجد مصادر", sources=[], cited_references=[],
        citation_status="not_applicable", generation=None,
    )
    assert response.generation is None
    payload = {
        "question": "سؤال", "answer": "جواب [S1]", "cited_references": ["S1"],
        "citation_status": "valid",
        "generation": {"model": "test", "finish_reason": "STOP"},
        "sources": [{"reference": "S1", "id": "p1", "score": 0.7, "cited": True,
                     "passage": {"id": "p1", "document_id": "d1", "chunk_index": 0,
                                 "text": "نص", "text_field": "text", "start": 0,
                                 "end": 2, "word_count": 1, "metadata": {"class": "civil"}}}],
    }
    assert RAGResponse.model_validate(payload).model_dump()["sources"] == payload["sources"]


@pytest.mark.parametrize("model", [
    ClassificationRequest, ClassificationResponse, LexicalSearchRequest,
    LexicalSearchResponse, NERRequest, NERResponse, RAGRequest, RAGResponse,
])
def test_models_expose_json_schema(model):
    assert model.model_json_schema()["type"] == "object"

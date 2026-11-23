"""Orchestration RAG sans appel réseau ni modèle NLP."""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.rag.chunker import chunk_documents
from src.rag.service import InvalidCitationError, RAGService


def setup_service(text="Réponse [S2], puis [S1] et [S2]."):
    passages = chunk_documents([
        {"id": "a", "text": "نص قانوني", "class": "real_estate"},
        {"id": "b", "text": "نص آخر", "class": "family_inheritance"},
    ])
    hits = [{"id": p["id"], "score": 0.9 - i * 0.1, "passage": p}
            for i, p in enumerate(passages)]
    retriever = Mock()
    retriever.retrieve.return_value = hits
    generator = Mock()
    generator.generate.return_value = SimpleNamespace(
        text=text, model="test-flash", finish_reason="STOP",
        prompt_tokens=20, output_tokens=10, total_tokens=30,
    )
    return RAGService(retriever, generator), hits


def test_pipeline_filters_prompt_citations_and_usage():
    service, hits = setup_service()
    before = deepcopy(hits)
    question = "  ما الحكم؟\n"
    result = service.answer(question, top_k=2, class_name="real_estate", score_threshold=0.3)
    service.retriever.retrieve.assert_called_once_with(
        question, top_k=2, document_id=None, class_name="real_estate", score_threshold=0.3,
    )
    service.generator.generate.assert_called_once()
    prompt = service.generator.generate.call_args.args[0]
    data = json.loads(prompt.split("(JSON) :\n", 1)[1])
    assert data["question"] == result["question"] == question
    assert [s["passage_id"] for s in data["sources"]] == [h["id"] for h in hits]
    assert result["cited_references"] == ["S2", "S1"]
    assert result["citation_status"] == "valid"
    assert result["generation"]["model"] == "test-flash"
    assert result["generation"]["total_tokens"] == 30
    for index, source in enumerate(result["sources"]):
        assert source["reference"] == data["sources"][index]["reference"]
        assert source["passage"] == hits[index]["passage"]
        assert source["score"] == hits[index]["score"]
        assert source["cited"]
    result["sources"][0]["passage"]["metadata"]["class"] = "changed"
    assert hits == before


@pytest.mark.parametrize("question", ["ما الحكم؟", "Quelle règle ?"])
def test_no_sources_skips_generation(question):
    service, _ = setup_service()
    service.retriever.retrieve.return_value = []
    result = service.answer(question)
    assert result["answer"]
    assert result["sources"] == result["cited_references"] == []
    assert result["citation_status"] == "not_applicable"
    assert result["generation"] is None
    service.generator.generate.assert_not_called()


@pytest.mark.parametrize("text", ["Réponse [S3]", "Réponse [S0]", "Réponse [S01]"])
def test_unknown_citations_rejected(text):
    service, _ = setup_service(text)
    with pytest.raises(InvalidCitationError):
        service.answer("Question")


def test_uncited_sources_are_not_reported_as_cited():
    service, _ = setup_service("Réponse [S2].")
    result = service.answer("Question")
    assert [s["cited"] for s in result["sources"]] == [False, True]
    assert result["cited_references"] == ["S2"]


def test_abstention_without_citation_is_explicit():
    service, _ = setup_service("Les passages ne permettent pas de répondre.")
    result = service.answer("Question")
    assert result["citation_status"] == "missing"
    assert result["cited_references"] == []
    assert not any(source["cited"] for source in result["sources"])


@pytest.mark.parametrize("component", ["retriever", "generator"])
def test_errors_are_propagated(component):
    service, _ = setup_service()
    error = RuntimeError("Échec technique")
    method = service.retriever.retrieve if component == "retriever" else service.generator.generate
    method.side_effect = error
    with pytest.raises(RuntimeError) as caught:
        service.answer("Question")
    assert caught.value is error
    if component == "retriever":
        service.generator.generate.assert_not_called()


def test_invalid_source_stops_before_generation():
    service, hits = setup_service()
    hits[0]["passage"]["text"] = ""
    with pytest.raises(ValueError):
        service.answer("Question")
    service.generator.generate.assert_not_called()


@pytest.mark.parametrize("question,error", [(None, TypeError), ("", ValueError), (" \n", ValueError)])
def test_invalid_question(question, error):
    service, _ = setup_service()
    with pytest.raises(error):
        service.answer(question)
    service.retriever.retrieve.assert_not_called()
    service.generator.generate.assert_not_called()

"""Ingestion et chaîne RAG intégrée, sans modèle externe ni appel réseau.

Qdrant, le retriever, le prompt et le wrapper Gemini sont réels. Seuls les
embeddings et le SDK distant sont simulés : ces tests vérifient le câblage et
les contrats, pas la qualité sémantique d'AraBERT ou des réponses de Gemini.
"""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from google.genai import errors, types

from src.common.arabic_normalizer import NormalizationConfig
from src.common.document_loader import CLASSES, load_documents
from src.rag import gemini_client
from src.rag.chunker import chunk_documents
from src.rag.gemini_client import (
    GeminiBlockedError,
    GeminiClient,
    GeminiRequestError,
    GeminiResponseError,
    GeminiTruncatedError,
)
from src.rag.ingestion import load_rag_documents, prepare_documents
from src.rag.qdrant_store import QdrantStore
from src.rag.retriever import RAGRetriever
from src.rag.service import InvalidCitationError, RAGService


def test_preserves_source_metadata_and_legal_text():
    text = "\ufeff  إِنَّ المحكمة لا تقبل ١٢/٣.\r\n\r\nArticle 4\rنص  "
    source = {"id": "a", "text": text, "metadata": {"tags": ["law"]}}
    before = deepcopy(source)
    result = prepare_documents(iter([source]))[0]
    assert result["prepared_text"] == "إِنَّ المحكمة لا تقبل ١٢/٣.\n\nArticle 4\nنص"
    assert result["text"] == text
    result["metadata"]["tags"].append("changed")
    assert source == before


def test_custom_normalization_and_repeated_preparation():
    documents = [{"id": "a", "text": " إِنَّ الـمَحْكَمَةَ\nلا تقبل ١٢. "}]
    config = NormalizationConfig(normalize_whitespace=True)
    prepared = prepare_documents(documents, normalization=config)
    assert prepared[0]["prepared_text"] == "ان المحكمة لا تقبل ١٢."
    assert prepare_documents(prepared, normalization=config) == prepared


def test_skips_empty_texts_with_warning_and_keeps_order(caplog):
    documents = [
        {"id": "z", "text": "same"},
        {"id": "empty", "text": "\ufeff \r\n\t"},
        {"id": "a", "text": "same"},
    ]
    assert [doc["id"] for doc in prepare_documents(documents)] == ["z", "a"]
    assert "empty" in caplog.text
    assert prepare_documents([]) == []
    assert prepare_documents(
        [{"id": "diacritics", "text": "َـ"}], normalization=NormalizationConfig()
    ) == []


@pytest.mark.parametrize("document", [{}, {"id": ""}, {"id": " "}, {"id": 3}])
def test_invalid_ids(document):
    with pytest.raises(ValueError, match="id"):
        prepare_documents([document])


@pytest.mark.parametrize("document", [None, "text", {"id": "a"}, {"id": "a", "text": 3}])
def test_invalid_documents(document):
    with pytest.raises(TypeError):
        prepare_documents([document])


def test_rejects_duplicate_ids_even_if_empty():
    with pytest.raises(ValueError, match="dupliqué"):
        prepare_documents([{"id": "a", "text": ""}, {"id": "a", "text": "valid"}])


def test_invalid_normalization():
    with pytest.raises(TypeError, match="normalization"):
        prepare_documents([], normalization={})


def test_loads_corpus_with_stable_metadata_and_strict_errors(tmp_path, caplog):
    for name in CLASSES:
        (tmp_path / name).mkdir()
    source = tmp_path / CLASSES[0] / "law.txt"
    source.write_text(" إِنَّ المحكمة ", encoding="utf-8")
    original_bytes = source.read_bytes()
    documents = load_rag_documents(tmp_path)
    assert documents == prepare_documents(load_documents(tmp_path))
    assert documents[0]["filename"] == "law.txt"
    assert documents[0]["class"] == CLASSES[0]
    assert documents[0]["path"] == str(source.resolve())
    assert source.read_bytes() == original_bytes
    (tmp_path / CLASSES[1] / "broken.txt").write_bytes(b"\xff")
    with pytest.raises(UnicodeDecodeError):
        load_rag_documents(tmp_path)
    assert load_rag_documents(tmp_path, strict=False) == documents
    assert "broken.txt" in caplog.text


def test_missing_class_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_rag_documents(tmp_path, strict=False)


@pytest.fixture
def rag_corpus(tmp_path):
    root = tmp_path / "corpus"
    for name in CLASSES:
        (root / name).mkdir(parents=True)
    texts = {
        "family_inheritance/family.txt": "نص تجريبي عن الأسرة والميراث.",
        "real_estate/property.txt": "\ufeff  إِنَّ المحكمة\nتبحث حقوق الملكية.  ",
        "real_estate/contract.txt": "نص تجريبي عن عقد العقار.",
        "real_estate/empty.txt": " \n\t",
    }
    for relative, text in texts.items():
        (root / relative).write_text(text, encoding="utf-8")
    return root


def _gemini_response(text="جواب تجريبي [S1] ثم [S2].", reason="STOP"):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(
            finish_reason=reason, content=types.Content(parts=[types.Part(text=text)]),
        )],
        model_version="test-flash",
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=40, candidates_token_count=12, total_token_count=52,
        ),
    )


@pytest.fixture
def rag_pipeline(rag_corpus, tmp_path, monkeypatch):
    # La fabrique est remplacée avant toute création du vrai wrapper Gemini.
    sdk = Mock()
    sdk.models.generate_content.return_value = _gemini_response()
    factory = Mock(return_value=sdk)
    monkeypatch.setattr(gemini_client.genai, "Client", factory)
    documents = load_rag_documents(rag_corpus)
    passages = chunk_documents(documents, chunk_size=20, overlap=0)
    vectors = {"property.txt": [1., 0.], "contract.txt": [0.8, 0.6], "family.txt": [0., 1.]}
    encoder = Mock(dimension=2)
    encoder.encode_query.return_value = np.array([1., 0.], dtype=np.float32)
    with QdrantStore(tmp_path / "qdrant", dimension=2) as store:
        store.upsert(passages, [vectors[p["metadata"]["filename"]] for p in passages])
        retriever = RAGRetriever(encoder, store)
        with GeminiClient(api_key="fake-key", model="test-flash") as client:
            yield SimpleNamespace(
                documents=documents, passages=passages, store=store, retriever=retriever,
                encoder=encoder, sdk=sdk, factory=factory, service=RAGService(retriever, client),
            )


def test_ingestion_to_passages_preserves_sources(rag_corpus, caplog):
    original_bytes = {p: p.read_bytes() for p in rag_corpus.rglob("*.txt")}
    documents = load_rag_documents(rag_corpus)
    assert len(documents) == 3
    assert "Document vide ignoré" in caplog.text
    assert {d["class"] for d in documents} == set(CLASSES)
    assert documents == load_rag_documents(rag_corpus)
    passages = chunk_documents(documents, chunk_size=3, overlap=1)
    by_id = {d["id"]: d for d in documents}
    for passage in passages:
        document = by_id[passage["document_id"]]
        assert passage["text_field"] == "prepared_text"
        assert passage["text"] == document["prepared_text"][passage["start"]:passage["end"]]
        assert passage["metadata"]["filename"] == document["filename"]
        assert passage["metadata"]["class"] == document["class"]
    assert {p: p.read_bytes() for p in original_bytes} == original_bytes


def test_retrieval_top_k_scores_and_filters(rag_pipeline):
    pipeline = rag_pipeline
    hits = pipeline.retriever.retrieve("حقوق الملكية", top_k=2)
    assert [h["passage"]["metadata"]["filename"] for h in hits] == ["property.txt", "contract.txt"]
    np.testing.assert_allclose([h["score"] for h in hits], [1., 0.8], atol=1e-6)
    assert len(pipeline.retriever.retrieve("سؤال", top_k=10)) == 3
    assert len(pipeline.retriever.retrieve("سؤال", score_threshold=0.9)) == 1
    family = pipeline.retriever.retrieve("سؤال", class_name="family_inheritance")
    assert len(family) == 1
    assert family[0]["passage"]["metadata"]["filename"] == "family.txt"
    document_id = family[0]["passage"]["document_id"]
    assert pipeline.retriever.retrieve("سؤال", document_id=document_id) == family
    assert pipeline.retriever.retrieve("سؤال", document_id=document_id, class_name="real_estate") == []
    pipeline.sdk.models.generate_content.assert_not_called()


def test_generation_uses_retrieved_sources_and_resolves_references(rag_pipeline):
    pipeline = rag_pipeline
    originals = deepcopy(pipeline.passages)
    question = "  ما هي حقوق الملكية؟\n"
    result = pipeline.service.answer(question, top_k=2, class_name="real_estate")
    pipeline.encoder.encode_query.assert_called_once_with(question)
    pipeline.sdk.models.generate_content.assert_called_once()
    call = pipeline.sdk.models.generate_content.call_args.kwargs
    assert call["model"] == "test-flash"
    data = json.loads(call["contents"].split("(JSON) :\n", 1)[1])
    assert data["question"] == result["question"] == question
    assert [s["filename"] for s in data["sources"]] == ["property.txt", "contract.txt"]
    assert result["answer"] == "جواب تجريبي [S1] ثم [S2]."
    assert result["cited_references"] == ["S1", "S2"]
    assert result["citation_status"] == "valid"
    assert result["generation"]["total_tokens"] == 52
    for source, prompt_source in zip(result["sources"], data["sources"]):
        assert source["reference"] == prompt_source["reference"]
        assert source["id"] == prompt_source["passage_id"]
        assert source["passage"]["text"] == prompt_source["text"]
        assert source["cited"]
    assert pipeline.passages == originals
    assert pipeline.store.count() == 3


@pytest.mark.parametrize("empty_index", [False, True])
def test_no_sources_never_calls_gemini(rag_pipeline, empty_index):
    pipeline = rag_pipeline
    if empty_index:
        for document in pipeline.documents:
            pipeline.store.delete_document(document["id"])
        options = {}
    else:
        options = {"document_id": "absent"}
    result = pipeline.service.answer("ما الحكم؟", **options)
    assert result["answer"]
    assert result["sources"] == result["cited_references"] == []
    assert result["citation_status"] == "not_applicable"
    assert result["generation"] is None
    pipeline.sdk.models.generate_content.assert_not_called()


@pytest.mark.parametrize("reason,text,error", [
    ("SAFETY", "", GeminiBlockedError),
    ("MAX_TOKENS", "جواب غير مكتمل [S1]", GeminiTruncatedError),
    ("STOP", " \n", GeminiResponseError),
])
def test_service_propagates_invalid_gemini_responses(rag_pipeline, reason, text, error):
    rag_pipeline.sdk.models.generate_content.return_value = _gemini_response(text, reason)
    with pytest.raises(error):
        rag_pipeline.service.answer("سؤال", top_k=1)
    assert rag_pipeline.store.count() == 3


def test_service_propagates_gemini_api_failure(rag_pipeline):
    rag_pipeline.sdk.models.generate_content.side_effect = errors.APIError(
        503, {"error": {"message": "simulated unavailable service"}},
    )
    with pytest.raises(GeminiRequestError) as caught:
        rag_pipeline.service.answer("سؤال")
    assert caught.value.status_code == 503
    assert caught.value.retryable


def test_service_rejects_fabricated_reference(rag_pipeline):
    rag_pipeline.sdk.models.generate_content.return_value = _gemini_response("جواب [S2]")
    with pytest.raises(InvalidCitationError, match="S2"):
        rag_pipeline.service.answer("سؤال", top_k=1)


def test_uncited_answer_is_identified(rag_pipeline):
    rag_pipeline.sdk.models.generate_content.return_value = _gemini_response("المقاطع غير كافية.")
    result = rag_pipeline.service.answer("سؤال", top_k=1)
    assert result["citation_status"] == "missing"
    assert result["cited_references"] == []
    assert not result["sources"][0]["cited"]


@pytest.mark.parametrize("question,options,error", [
    (None, {}, TypeError), (" \n", {}, ValueError),
    ("سؤال", {"top_k": 0}, ValueError), ("سؤال", {"score_threshold": float("nan")}, ValueError),
])
def test_invalid_request_stops_before_encoding_and_generation(rag_pipeline, question, options, error):
    with pytest.raises(error):
        rag_pipeline.service.answer(question, **options)
    rag_pipeline.encoder.encode_query.assert_not_called()
    rag_pipeline.sdk.models.generate_content.assert_not_called()

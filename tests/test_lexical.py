"""Tests unitaires lexicaux et recherche sur un corpus synthétique arabe."""

from copy import deepcopy
from math import log, sqrt

import numpy as np
import pytest

from src.common.arabic_normalizer import NormalizationConfig
from src.common.document_loader import load_documents
from src.common.document_schema import Document
from src.lexical_search.chi2_selector import compute_chi2_scores, select_discriminant_terms
from src.lexical_search.cosine_search import cosine_similarities, rank_top_k
from src.lexical_search.indexer import build_inverted_index
from src.lexical_search.preprocessing import (
    PreprocessingConfig,
    clean_text,
    preprocess_text,
    remove_stopwords,
    stem_tokens,
    tokenize,
)
from src.lexical_search.service import LexicalSearchService
from src.lexical_search.tfidf import TfidfModel


def test_cleaning_preserves_word_boundaries_and_arabic_letters():
    assert clean_text("حكم،قانون! English\tعقد🙂ملكية") == "حكم قانون عقد ملكية"
    assert clean_text("ال\u200dعقد") == "العقد"
    assert clean_text("ﻻ") == "لا"


def test_tokenization_preserves_diacritics_repetitions_and_numbers():
    assert tokenize("عَقْد،عَقْد المادة12 ١٢ ۱۲") == ["عَقْد", "عَقْد", "المادة", "12", "١٢", "۱۲"]
    assert tokenize("عقد12ملكية ١٢", keep_numbers=False) == ["عقد", "ملكية"]
    assert tokenize("َُ 123") == ["123"]


def test_pipeline_normalizes_and_removes_stopwords_without_mutating_source():
    document = {"text": "إِنَّ الـمَحْكَمَةَ في العقار، العقار ١٢"}
    before = document.copy()
    config = PreprocessingConfig(stemming=False)

    assert preprocess_text(document["text"], config) == ["المحكمة", "العقار", "العقار", "١٢"]
    assert document == before


def test_stopwords_use_the_same_normalization_as_text():
    config = PreprocessingConfig(
        stemming=False,
        normalization=NormalizationConfig(normalize_alef_maqsura=True),
    )
    assert preprocess_text("إِلَى على المحكمة", config) == ["المحكمة"]
    assert preprocess_text("عَلَى المحكمة", config, stopwords=["عَلَى"]) == ["المحكمة"]


def test_custom_and_disabled_stopwords():
    config = PreprocessingConfig(stemming=False)
    assert preprocess_text("في حكم", config, stopwords=[]) == ["في", "حكم"]
    assert preprocess_text("في حكم", config, stopwords=["حكم"]) == ["في"]
    config = PreprocessingConfig(stemming=False, remove_stopwords=False)
    assert preprocess_text("في حكم", config) == ["في", "حكم"]
    tokens = ["في", "حكم", "حكم"]
    assert remove_stopwords(tokens) == ["حكم", "حكم"]
    assert tokens == ["في", "حكم", "حكم"]


def test_negations_are_retained():
    assert preprocess_text("لا لم لن ليس", PreprocessingConfig(stemming=False)) == [
        "لا", "لم", "لن", "ليس"
    ]


def test_stemming_reduces_definite_article_and_preserves_numbers():
    tokens = ["الكتاب", "كتاب", "١٢", "12"]
    assert stem_tokens(tokens) == ["كتاب", "كتاب", "١٢", "12"]
    assert tokens == ["الكتاب", "كتاب", "١٢", "12"]
    assert preprocess_text("الكتاب كتاب") == ["كتاب", "كتاب"]


def test_number_and_length_options():
    config = PreprocessingConfig(stemming=False, keep_numbers=False, min_token_length=2)
    assert preprocess_text("س عقد ١٢", config) == ["عقد"]
    config = PreprocessingConfig(
        stemming=False, normalization=NormalizationConfig(normalize_digits=True)
    )
    assert preprocess_text("١٢ ۱۲", config) == ["12", "12"]


@pytest.mark.parametrize("text", ["", " \n\t", "!!!🙂", "English only", "في من إلى"])
def test_empty_or_filtered_text(text):
    assert preprocess_text(text) == []


def test_invalid_inputs():
    with pytest.raises(TypeError, match="text"):
        preprocess_text(None)
    with pytest.raises(TypeError, match="text"):
        clean_text(None)
    with pytest.raises(ValueError, match="min_token_length"):
        PreprocessingConfig(min_token_length=0)
    with pytest.raises(TypeError, match="min_token_length"):
        PreprocessingConfig(min_token_length=1.5)


def make_document(document_id: str, text: str, class_name: str = "real_estate") -> Document:
    return {
        "id": document_id,
        "class": class_name,
        "filename": "same.txt",
        "path": f"{class_name}/same.txt",
        "text": text,
    }


def test_index_counts_terms_per_document_without_mutation():
    documents = [
        make_document("doc-1", "عَقْد عقد ملكية في"),
        make_document("doc-2", "عقد إرث إرث", "family_inheritance"),
    ]
    original = deepcopy(documents)

    index = build_inverted_index(documents, PreprocessingConfig(stemming=False))

    assert index == {
        "عقد": {"doc-1": 2, "doc-2": 1},
        "ملكية": {"doc-1": 1},
        "ارث": {"doc-2": 2},
    }
    assert documents == original


def test_index_uses_stemming_by_default():
    assert build_inverted_index([make_document("doc", "الكتاب كتاب")]) == {
        "كتاب": {"doc": 2}
    }


def test_index_handles_empty_corpus_and_documents_without_terms():
    assert build_inverted_index([]) == {}
    assert build_inverted_index([
        make_document("empty", ""),
        make_document("stopwords", "في من إلى"),
        make_document("symbols", "!!!"),
    ]) == {}


def test_index_supports_generators_and_reuses_custom_stopwords():
    documents = (make_document(str(number), "عقد ملكية") for number in range(2))
    stopwords = (word for word in ["عقد"])
    assert build_inverted_index(
        documents, PreprocessingConfig(stemming=False), stopwords=stopwords
    ) == {"ملكية": {"0": 1, "1": 1}}


def test_index_order_is_independent_of_document_order():
    documents = [make_document("z", "ملكية عقد"), make_document("a", "عقد إرث")]
    config = PreprocessingConfig(stemming=False)
    forward = build_inverted_index(documents, config)
    backward = build_inverted_index(reversed(documents), config)

    assert list(forward.items()) == list(backward.items())
    assert list(forward) == sorted(forward)
    assert list(forward["عقد"]) == ["a", "z"]
    assert list(backward["عقد"]) == ["a", "z"]


@pytest.mark.parametrize("text", ["", "عقد"])
def test_index_rejects_duplicate_ids_even_for_empty_documents(text):
    documents = [make_document("same", text), make_document("same", "ملكية")]
    with pytest.raises(ValueError, match="dupliqué"):
        build_inverted_index(documents)


@pytest.mark.parametrize("document_id", ["", "   "])
def test_index_rejects_empty_ids(document_id):
    with pytest.raises(ValueError, match="vide"):
        build_inverted_index([make_document(document_id, "عقد")])


def test_index_rejects_non_string_ids():
    with pytest.raises(TypeError, match="identifiant"):
        build_inverted_index([make_document(1, "عقد")])


def test_index_propagates_invalid_text_error():
    with pytest.raises(TypeError, match="text"):
        build_inverted_index([make_document("doc", None)])


@pytest.fixture
def synthetic_corpus():
    """Deux documents immobiliers, quatre successoraux, avec un terme partagé."""
    return [
        make_document("r1", "عَقْد عقد ملكية حكم"),
        make_document("f1", "عقد إرث حكم", "family_inheritance"),
        make_document("r2", "ملكية حكم"),
        make_document("f2", "إرث حكم", "family_inheritance"),
        make_document("f3", "إرث حكم", "family_inheritance"),
        make_document("f4", "إرث حكم", "family_inheritance"),
    ]


@pytest.fixture
def synthetic_service(synthetic_corpus):
    # Sans stemming : les valeurs attendues isolent les calculs statistiques.
    return LexicalSearchService(PreprocessingConfig(stemming=False)).fit(synthetic_corpus)


def test_chi2_on_unbalanced_synthetic_corpus(synthetic_corpus):
    scores = compute_chi2_scores(synthetic_corpus, PreprocessingConfig(stemming=False))
    # Pour عقد : A=1, B=1, C=1, D=3, donc 6*(3-1)^2/(2*4*2*4) = 0.375.
    for label in ("real_estate", "family_inheritance"):
        assert scores[label]["عقد"] == pytest.approx(0.375)
        assert scores[label]["ملكية"] == pytest.approx(6)
        assert scores[label]["ارث"] == pytest.approx(6)
        assert scores[label]["حكم"] == 0
    selected = select_discriminant_terms(scores, k=None, min_score=0.375)
    assert selected == {
        "family_inheritance": ["ارث", "ملكية"],
        "real_estate": ["ارث", "ملكية"],
    }


def test_tfidf_and_cosine_match_manual_calculation(synthetic_corpus):
    model = TfidfModel(PreprocessingConfig(stemming=False), norm=None)
    matrix = model.fit_transform(synthetic_corpus)
    query = model.vectorize_query("عقد")
    contract_column = model.vocabulary_["عقد"]
    idf = log(7 / 3) + 1
    inheritance_idf = log(7 / 5) + 1

    assert model.idf_[contract_column] == pytest.approx(idf)
    assert matrix[0, contract_column] == pytest.approx(2 * idf)
    assert query[0, contract_column] == pytest.approx(idf)
    expected = [
        2 * idf / sqrt(5 * idf**2 + 1),
        idf / sqrt(idf**2 + inheritance_idf**2 + 1),
        0, 0, 0, 0,
    ]
    scores = cosine_similarities(query, matrix)
    np.testing.assert_allclose(scores, expected)
    assert [hit["id"] for hit in rank_top_k(scores, model.document_ids_)] == ["r1", "f1"]


def test_class_choice_uses_proportions_not_raw_document_counts(synthetic_service):
    # عقد apparaît dans un document de chaque classe, mais dans 1/2 contre 1/4.
    response = synthetic_service.search("عقد")
    assert response["selected_class"] == "real_estate"
    assert response["class_scores"] == {"family_inheritance": 0.0, "real_estate": 0.375}
    assert [hit["id"] for hit in response["results"]] == ["r1"]


@pytest.mark.parametrize(
    ("query", "expected_class", "expected_ids"),
    [
        ("مِلْكِيَّة", "real_estate", ["r2", "r1"]),
        ("إِرْث", "family_inheritance", ["f2", "f3"]),
    ],
)
def test_synthetic_search_class_and_top_k(synthetic_service, query, expected_class, expected_ids):
    response = synthetic_service.search(query, k=2)
    assert response["selected_class"] == expected_class
    assert [hit["id"] for hit in response["results"]] == expected_ids
    assert all(hit["class"] == expected_class for hit in response["results"])
    scores = [hit["score"] for hit in response["results"]]
    assert scores == sorted(scores, reverse=True)


def test_common_terms_search_all_classes(synthetic_service):
    response = synthetic_service.search("حكم", k=10)
    assert response["selected_class"] is None
    assert response["class_scores"] == {"family_inheritance": 0.0, "real_estate": 0.0}
    assert len(response["results"]) == 6
    assert {hit["class"] for hit in response["results"]} == {
        "family_inheritance", "real_estate"
    }


def test_repeated_query_terms_do_not_inflate_class_score(synthetic_service):
    single = synthetic_service.search("عقد")
    repeated = synthetic_service.search("عقد عقد عقد")
    assert repeated["class_scores"] == single["class_scores"]
    assert repeated["selected_class"] == single["selected_class"]
    assert [hit["id"] for hit in repeated["results"]] == ["r1"]
    assert repeated["results"][0]["score"] == pytest.approx(single["results"][0]["score"])


def test_query_order_does_not_change_search_statistics(synthetic_service):
    before = synthetic_service.search("عقد")
    synthetic_service.search("إرث")
    synthetic_service.search("مجهول")
    assert synthetic_service.search("عقد") == before


def test_corpus_order_does_not_change_search_results(synthetic_corpus, synthetic_service):
    reversed_service = LexicalSearchService(PreprocessingConfig(stemming=False)).fit(
        reversed(synthetic_corpus)
    )
    for query in ("عقد", "إرث", "حكم"):
        expected = synthetic_service.search(query)
        actual = reversed_service.search(query)
        assert actual["selected_class"] == expected["selected_class"]
        assert actual["class_scores"] == expected["class_scores"]
        assert [hit["id"] for hit in actual["results"]] == [
            hit["id"] for hit in expected["results"]
        ]
        np.testing.assert_allclose(
            [hit["score"] for hit in actual["results"]],
            [hit["score"] for hit in expected["results"]],
        )


def test_utf8_files_to_search_preserve_original_text(synthetic_corpus, tmp_path):
    original_files = {}
    for document in synthetic_corpus:
        folder = tmp_path / document["class"]
        folder.mkdir(exist_ok=True)
        path = folder / f"{document['id']}.txt"
        path.write_text(document["text"], encoding="utf-8")
        original_files[path] = path.read_bytes()
    documents = load_documents(tmp_path)
    original_documents = deepcopy(documents)
    service = LexicalSearchService(PreprocessingConfig(stemming=False)).fit(documents)

    response = service.search("عَقْد", k=1)

    assert response["results"][0]["filename"] == "r1.txt"
    assert response["results"][0]["text"] == "عَقْد عقد ملكية حكم"
    assert documents == original_documents
    assert all(path.read_bytes() == content for path, content in original_files.items())

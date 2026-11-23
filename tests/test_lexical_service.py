"""Tests de la chaîne de recherche et de l'orientation par classe."""

from copy import deepcopy

import pytest
from sklearn.exceptions import NotFittedError

from src.lexical_search.preprocessing import PreprocessingConfig
from src.lexical_search.service import LexicalSearchService

CONFIG = PreprocessingConfig(stemming=False)


@pytest.fixture
def documents():
    return [
        {"id": doc_id, "class": label, "filename": f"{doc_id}.txt",
         "path": f"{label}/{doc_id}.txt", "text": text}
        for doc_id, label, text in [
            ("r2", "real_estate", "عَقْد حكم"),
            ("f2", "family_inheritance", "إرث حكم"),
            ("r1", "real_estate", "عَقْد حكم"),
            ("f1", "family_inheritance", "إرث حكم"),
        ]
    ]


def test_binary_class_choice_uses_positive_associations(documents):
    service = LexicalSearchService(CONFIG).fit(iter(documents))
    assert service.chi2_scores_["real_estate"] == service.chi2_scores_["family_inheritance"]
    assert service.selected_terms_ == {"family_inheritance": ["ارث"], "real_estate": ["عقد"]}

    for query, expected_class, expected_ids in [
        ("عَقْد", "real_estate", ["r1", "r2"]),
        ("إرث", "family_inheritance", ["f1", "f2"]),
    ]:
        response = service.search(query, k=10)
        assert response["selected_class"] == expected_class
        assert response["class_scores"][expected_class] == pytest.approx(4)
        assert [hit["id"] for hit in response["results"]] == expected_ids
        assert all(hit["class"] == expected_class for hit in response["results"])
        assert all(0 < hit["score"] <= 1 for hit in response["results"])


@pytest.mark.parametrize("query", ["حكم", "عقد إرث"])
def test_no_signal_or_tie_searches_whole_corpus(documents, query):
    response = LexicalSearchService(CONFIG).fit(documents).search(query)
    assert response["selected_class"] is None
    assert len(response["results"]) == 4


def test_forced_class_top_k_and_threshold(documents):
    service = LexicalSearchService(CONFIG).fit(documents)
    response = service.search("حكم", class_name="family_inheritance", k=1)
    assert response["selected_class"] == "family_inheritance"
    assert [hit["id"] for hit in response["results"]] == ["f1"]
    assert service.search("عقد", class_name="family_inheritance")["results"] == []
    assert service.search("عقد", k=0)["results"] == []
    assert service.search("عقد", min_score=1)["results"] == []


@pytest.mark.parametrize("query", ["", "في من", "مجهول"])
def test_empty_filtered_or_unknown_query(documents, query):
    response = LexicalSearchService(CONFIG).fit(documents).search(query)
    assert response["selected_class"] is None
    assert response["results"] == []


def test_original_texts_and_returned_copies(documents):
    before = deepcopy(documents)
    service = LexicalSearchService(CONFIG).fit(documents)
    response = service.search("عقد")
    assert documents == before
    assert response["results"][0]["text"] == "عَقْد حكم"
    response["results"][0]["text"] = "changed"
    documents[0]["text"] = "changed input"
    selected = service.selected_terms_
    selected["real_estate"].clear()
    assert service.search("عقد")["results"][0]["text"] == "عَقْد حكم"
    assert service.selected_terms_["real_estate"] == ["عقد"]


def test_selection_limit_and_custom_stopwords(documents):
    service = LexicalSearchService(CONFIG, chi2_k=0).fit(documents)
    assert service.search("عقد")["selected_class"] is None
    assert len(service.search("عقد")["results"]) == 2
    service = LexicalSearchService(CONFIG, stopwords=iter(["عقد"])).fit(documents)
    assert service.search("عقد")["tokens"] == []
    assert service.search("عقد")["results"] == []


def test_default_stemming_end_to_end(documents):
    documents[0]["text"] = documents[2]["text"] = "الكتاب حكم"
    response = LexicalSearchService().fit(documents).search("كتاب")
    assert response["selected_class"] == "real_estate"
    assert [hit["id"] for hit in response["results"]] == ["r1", "r2"]


def test_errors_and_failed_refit_preserve_previous_state(documents):
    service = LexicalSearchService(CONFIG)
    with pytest.raises(NotFittedError):
        service.search("عقد")
    service.fit(documents)
    before = service.search("عقد")
    with pytest.raises(ValueError):
        service.fit([])
    assert service.search("عقد") == before
    with pytest.raises(ValueError, match="Classe inconnue"):
        service.search("عقد", class_name="unknown")
    with pytest.raises(ValueError, match="k"):
        service.search("", k=-1)
    with pytest.raises(TypeError):
        service.search(None)

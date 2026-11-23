"""Tables de contingence connues et sélection déterministe des termes."""

from copy import deepcopy

import pytest

from src.lexical_search.chi2_selector import compute_chi2_scores, select_discriminant_terms
from src.lexical_search.preprocessing import PreprocessingConfig

CONFIG = PreprocessingConfig(stemming=False, remove_stopwords=False)


def make_documents(rows):
    return [
        {"id": str(index), "class": label, "filename": f"{index}.txt",
         "path": f"{label}/{index}.txt", "text": text}
        for index, (label, text) in enumerate(rows)
    ]


def test_perfect_association_and_ubiquitous_term():
    documents = make_documents([
        ("real_estate", "عَقْد عقد حكم"), ("real_estate", "عقد حكم"),
        ("family_inheritance", "إرث حكم"), ("family_inheritance", "إرث حكم"),
    ])
    original = deepcopy(documents)

    scores = compute_chi2_scores(iter(documents), CONFIG)

    assert scores == {
        "family_inheritance": {"ارث": 4.0, "حكم": 0.0, "عقد": 4.0},
        "real_estate": {"ارث": 4.0, "حكم": 0.0, "عقد": 4.0},
    }
    assert documents == original


def test_independent_term_has_zero_score():
    documents = make_documents([("a", "عقد"), ("a", ""), ("b", "عقد"), ("b", "")])
    assert compute_chi2_scores(documents, CONFIG) == {"a": {"عقد": 0.0}, "b": {"عقد": 0.0}}


def test_multiclass_one_against_rest_includes_empty_documents():
    documents = make_documents([
        ("a", "عقد"), ("a", "عقد"), ("b", ""), ("b", ""), ("c", ""), ("c", ""),
    ])
    scores = compute_chi2_scores(documents, CONFIG)
    assert scores["a"]["عقد"] == pytest.approx(6.0)
    assert scores["b"]["عقد"] == pytest.approx(1.5)
    assert scores["c"]["عقد"] == pytest.approx(1.5)


def test_repetitions_do_not_change_document_presence_scores():
    documents = make_documents([("a", "عقد"), ("b", "ملكية"), ("b", "")])
    before = compute_chi2_scores(documents, CONFIG)
    documents[0]["text"] = "عقد " * 100
    assert compute_chi2_scores(documents, CONFIG) == before


def test_custom_stopwords_and_empty_vocabulary():
    documents = make_documents([("a", "عقد"), ("b", "عقد")])
    assert compute_chi2_scores(
        documents, PreprocessingConfig(stemming=False), stopwords=iter(["عقد"])
    ) == {"a": {}, "b": {}}
    assert select_discriminant_terms({"a": {}, "b": {}}) == {"a": [], "b": []}


def test_ranking_threshold_ties_and_limits():
    scores = {"b": {"z": 3.0, "a": 3.0, "best": 5.0, "zero": 0.0}, "a": {}}
    before = deepcopy(scores)
    assert select_discriminant_terms(scores, k=2) == {"a": [], "b": ["best", "a"]}
    assert select_discriminant_terms(scores, k=None)["b"] == ["best", "a", "z"]
    assert select_discriminant_terms(scores, min_score=3)["b"] == ["best"]
    assert select_discriminant_terms(scores, k=0) == {"a": [], "b": []}
    assert scores == before


def test_invalid_corpus():
    with pytest.raises(ValueError, match="vide"):
        compute_chi2_scores([])
    with pytest.raises(ValueError, match="deux classes"):
        compute_chi2_scores(make_documents([("a", "عقد")]))
    with pytest.raises(ValueError, match="classe non vide"):
        compute_chi2_scores(make_documents([("", "عقد"), ("b", "ملكية")]))
    documents = make_documents([("a", "عقد"), ("b", "ملكية")])
    documents[1]["id"] = documents[0]["id"]
    with pytest.raises(ValueError, match="dupliqué"):
        compute_chi2_scores(documents, CONFIG)


@pytest.mark.parametrize("k", [-1, 1.5, True])
def test_invalid_selection_limit(k):
    with pytest.raises(ValueError, match="k"):
        select_discriminant_terms({}, k=k)


@pytest.mark.parametrize("score", [-1, float("nan"), float("inf")])
def test_invalid_scores_and_thresholds(score):
    with pytest.raises(ValueError):
        select_discriminant_terms({}, min_score=score)
    with pytest.raises(ValueError):
        select_discriminant_terms({"a": {"term": score}})

"""Calcul cosinus, classement et intégration avec le modèle TF-IDF."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.lexical_search.cosine_search import cosine_similarities, rank_top_k, search_top_k
from src.lexical_search.preprocessing import PreprocessingConfig
from src.lexical_search.tfidf import TfidfModel


@pytest.mark.parametrize("sparse", [False, True])
def test_cosine_for_unnormalized_vectors_and_zero_rows(sparse):
    documents = np.array([[3, 4], [4, 0], [0, 0], [-3, -4]], dtype=float)
    query = np.array([[6, 8]], dtype=float)
    if sparse:
        documents, query = csr_matrix(documents), csr_matrix(query)
    before = documents.copy()

    np.testing.assert_allclose(cosine_similarities(query, documents), [1, 0.6, 0, -1])
    if sparse:
        assert (documents != before).nnz == 0
    else:
        np.testing.assert_array_equal(documents, before)


def test_empty_query_corpus_and_feature_space():
    np.testing.assert_array_equal(cosine_similarities([0, 0], [[1, 2], [0, 0]]), [0, 0])
    assert search_top_k([0, 0], [[1, 2]], ["doc"]) == []
    assert search_top_k([1, 2], csr_matrix((0, 2)), []) == []
    np.testing.assert_array_equal(cosine_similarities([], csr_matrix((2, 0))), [0, 0])


def test_ranking_threshold_and_deterministic_ties():
    scores = [0.5, 0.9, 0.9, 0.0]
    ids = ["middle", "z", "a", "zero"]
    assert rank_top_k(scores, iter(ids), k=2) == [
        {"id": "a", "score": 0.9}, {"id": "z", "score": 0.9}
    ]
    assert [hit["id"] for hit in rank_top_k(scores, ids, k=99)] == ["a", "z", "middle"]
    assert rank_top_k(scores, ids, k=0) == []
    assert rank_top_k(scores, ids, min_score=0.9) == []
    assert rank_top_k([0], ["zero"], min_score=-0.1) == [{"id": "zero", "score": 0.0}]
    assert rank_top_k([], []) == []


def test_search_with_tfidf_and_reloaded_model(tmp_path):
    documents = [
        {"id": doc_id, "class": "real_estate", "filename": f"{doc_id}.txt",
         "path": f"{doc_id}.txt", "text": text}
        for doc_id, text in [("b", "ملكية"), ("a", "عقد"), ("c", "عقد ملكية")]
    ]
    model = TfidfModel(PreprocessingConfig(stemming=False), norm=None)
    matrix = model.fit_transform(documents)
    path = tmp_path / "tfidf.joblib"
    model.save(path)
    loaded = TfidfModel.load(path)

    results = search_top_k(loaded.vectorize_query("عقد"), matrix, loaded.document_ids_, k=2)

    assert [hit["id"] for hit in results] == ["a", "c"]
    assert results[0]["score"] == pytest.approx(1)
    assert results[1]["score"] == pytest.approx(1 / np.sqrt(2))
    assert search_top_k(loaded.vectorize_query("مجهول"), matrix, loaded.document_ids_) == []


@pytest.mark.parametrize(
    ("query", "documents"),
    [
        ([1], [[1, 2]]),
        ([[1], [2]], [[1]]),
        ([1], [1]),
        ([float("nan")], [[1]]),
        ([1], [[float("inf")]]),
        ([1j], [[1]]),
    ],
)
def test_invalid_vectors(query, documents):
    with pytest.raises(ValueError):
        cosine_similarities(query, documents)


@pytest.mark.parametrize("k", [-1, 1.5, True])
def test_invalid_k(k):
    with pytest.raises(ValueError, match="k"):
        rank_top_k([1], ["doc"], k=k)


@pytest.mark.parametrize("ids", [[], ["a", "a"], ["", "b"], [1, "b"]])
def test_invalid_document_ids(ids):
    with pytest.raises(ValueError):
        rank_top_k([0.5, 0.8], ids)


@pytest.mark.parametrize("scores", [[float("nan")], [float("inf")], [1.1], [[0.5]]])
def test_invalid_scores(scores):
    with pytest.raises(ValueError):
        rank_top_k(scores, ["doc"])


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -2, 2])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        rank_top_k([0.5], ["doc"], min_score=threshold)

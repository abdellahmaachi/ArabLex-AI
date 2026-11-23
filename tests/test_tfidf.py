"""Valeurs TF-IDF attendues et réutilisation des statistiques sauvegardées."""

from copy import deepcopy
from math import log

import joblib
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from src.common.arabic_normalizer import NormalizationConfig
from src.lexical_search.preprocessing import PreprocessingConfig
from src.lexical_search.tfidf import TfidfModel


@pytest.fixture
def documents():
    return [
        {"id": doc_id, "class": "real_estate", "filename": f"{doc_id}.txt",
         "path": f"real_estate/{doc_id}.txt", "text": text}
        for doc_id, text in [("b", "عقد عقد ملكية"), ("a", "عقد"), ("empty", "")]
    ]


def test_raw_tfidf_formula_and_row_alignment(documents):
    original = deepcopy(documents)
    model = TfidfModel(PreprocessingConfig(stemming=False), norm=None)
    matrix = model.fit_transform(iter(documents))
    contract = model.vocabulary_["عقد"]
    ownership = model.vocabulary_["ملكية"]

    assert model.document_ids_ == ("b", "a", "empty")
    assert model.n_documents_ == 3
    assert matrix.format == "csr"
    assert matrix.shape == (3, 2)
    assert model.idf_[contract] == pytest.approx(log(4 / 3) + 1)
    assert model.idf_[ownership] == pytest.approx(log(4 / 2) + 1)
    assert model.document_frequency_[contract] == 2
    assert model.document_frequency_[ownership] == 1
    assert matrix[0, contract] == pytest.approx(2 * (log(4 / 3) + 1))
    assert matrix[0, ownership] == pytest.approx(log(2) + 1)
    assert matrix[1, contract] == pytest.approx(log(4 / 3) + 1)
    assert matrix[2].nnz == 0
    assert documents == original


def test_l2_normalization_and_query_use_corpus_idf(documents):
    model = TfidfModel(PreprocessingConfig(stemming=False))
    matrix = model.fit_transform(documents)
    np.testing.assert_allclose(np.linalg.norm(matrix.toarray(), axis=1), [1, 1, 0])
    vocabulary = model.vocabulary_
    before_idf = model.idf_
    query = model.vectorize_query("عقد ملكية مجهول")
    expected = before_idf / np.linalg.norm(before_idf)
    np.testing.assert_allclose(query.toarray()[0], expected)
    np.testing.assert_array_equal(model.idf_, before_idf)
    assert model.vocabulary_ == vocabulary
    assert model.n_documents_ == 3
    assert model.vectorize_query("مجهول").nnz == 0
    assert model.vectorize_query("").shape == (1, 2)
    assert model.vectorize_query("").nnz == 0
    assert model.transform([]).shape == (0, 2)


def test_saved_statistics_reproduce_queries_and_preprocessing(documents, tmp_path):
    documents[0]["text"] += " على"
    config = PreprocessingConfig(
        stemming=False, normalization=NormalizationConfig(normalize_alef_maqsura=True)
    )
    model = TfidfModel(config, stopwords=(word for word in ["ملكية"]), norm=None)
    corpus_matrix = model.fit_transform(documents)
    target = tmp_path / "nested" / "tfidf.joblib"
    model.save(target)
    loaded = TfidfModel.load(target)

    assert loaded.document_ids_ == model.document_ids_
    assert loaded.n_documents_ == model.n_documents_
    assert loaded.vocabulary_ == model.vocabulary_
    np.testing.assert_array_equal(loaded.idf_, model.idf_)
    np.testing.assert_array_equal(loaded.document_frequency_, model.document_frequency_)
    np.testing.assert_allclose(
        loaded.vectorize_query("عَقْد عقد ملكية علي").toarray(), corpus_matrix[0].toarray()
    )
    np.testing.assert_allclose(
        loaded.transform(doc["text"] for doc in documents).toarray(), corpus_matrix.toarray()
    )


def test_unfitted_model_is_rejected(tmp_path):
    model = TfidfModel()
    with pytest.raises(NotFittedError):
        model.vectorize_query("عقد")
    with pytest.raises(NotFittedError):
        model.save(tmp_path / "unfitted.joblib")


def test_invalid_corpus_and_failed_refit(documents):
    model = TfidfModel(PreprocessingConfig(stemming=False)).fit(documents)
    original = model.vectorize_query("عقد").toarray()
    with pytest.raises(ValueError, match="vide"):
        model.fit([])
    with pytest.raises(ValueError, match="aucun terme"):
        model.fit([dict(documents[0], text="في من")])
    with pytest.raises(ValueError, match="dupliqué"):
        model.fit([documents[0], documents[0]])
    np.testing.assert_array_equal(model.vectorize_query("عقد").toarray(), original)
    with pytest.raises(TypeError):
        model.transform("عقد")


def test_invalid_artifact(tmp_path):
    target = tmp_path / "invalid.joblib"
    joblib.dump({"version": 99}, target)
    with pytest.raises(ValueError, match="Format"):
        TfidfModel.load(target)

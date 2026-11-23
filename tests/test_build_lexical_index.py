"""Construction offline, rechargement et conservation de l'index actif."""

import json

import numpy as np
import pytest
from scipy.sparse import load_npz

from scripts.build_lexical_index import build_artifacts, main
from src.lexical_search.preprocessing import PreprocessingConfig
from src.lexical_search.service import LexicalSearchService
from src.lexical_search.tfidf import TfidfModel


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "raw"
    for label, text in [("real_estate", "عَقْد حكم"), ("family_inheritance", "إرث حكم")]:
        folder = root / label
        folder.mkdir(parents=True)
        (folder / "one.txt").write_text(text, encoding="utf-8")
    return root


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_all_artifacts_and_reload_without_fit(corpus, tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    target = build_artifacts(corpus, root, config=PreprocessingConfig(stemming=False))
    assert read_json(root / "current.json")["directory"] == target.name
    manifest = read_json(target / "manifest.json")
    assert all((target / name).is_file() for name in manifest["files"])
    assert manifest["class_counts"] == {"family_inheritance": 1, "real_estate": 1}
    assert manifest["preprocessing"]["stemming"] is False
    documents = read_json(target / "documents.json")
    statistics = read_json(target / "tfidf_statistics.json")
    assert [doc["id"] for doc in documents] == statistics["document_ids"]
    matrix = load_npz(target / "tfidf_matrix.npz")
    assert list(matrix.shape) == manifest["matrix_shape"]
    assert "عقد" in read_json(target / "inverted_index.json")

    def no_training(*args, **kwargs):
        raise AssertionError("Le rechargement ne doit pas réentraîner")

    monkeypatch.setattr(LexicalSearchService, "fit", no_training)
    monkeypatch.setattr(TfidfModel, "fit_transform", no_training)
    service = LexicalSearchService.load(target / "service.joblib")
    response = service.search("عقد")
    assert response["selected_class"] == "real_estate"
    assert response["results"][0]["text"] == "عَقْد حكم"
    model = TfidfModel.load(target / "tfidf_model.joblib")
    np.testing.assert_allclose(
        model.transform(doc["text"] for doc in documents).toarray(), matrix.toarray()
    )


def test_failed_build_preserves_current_index(corpus, tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    first = build_artifacts(corpus, root)
    pointer = (root / "current.json").read_bytes()

    def fail_save(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(LexicalSearchService, "save_artifacts", fail_save)
    with pytest.raises(OSError, match="disk full"):
        build_artifacts(corpus, root)
    assert (root / "current.json").read_bytes() == pointer
    assert (first / "service.joblib").is_file()


def test_rebuild_preserves_previous_generation(corpus, tmp_path):
    root = tmp_path / "artifacts"
    first = build_artifacts(corpus, root)
    second = build_artifacts(corpus, root)
    assert first != second
    assert (first / "manifest.json").is_file()
    assert read_json(root / "current.json")["directory"] == second.name


def test_cli_errors_and_strict_mode(corpus, tmp_path, capsys):
    root = tmp_path / "artifacts"
    (corpus / "real_estate" / "bad.txt").write_bytes(b"\xff")
    assert main(["--data-dir", str(corpus), "--output-dir", str(root), "--strict"]) == 1
    assert "Erreur de construction" in capsys.readouterr().err
    assert not (root / "current.json").exists()
    assert main([
        "--data-dir", str(corpus), "--output-dir", str(root), "--no-stemming"
    ]) == 0
    assert "Total : 2 documents" in capsys.readouterr().out

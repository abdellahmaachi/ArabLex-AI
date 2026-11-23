"""Pipeline hors ligne : vrai Qdrant Local et encodeur déterministe simulé."""

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from scripts import build_rag_index as pipeline
from src.common.document_loader import CLASSES
from src.rag.qdrant_store import QdrantStore


@pytest.fixture
def setup(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    for name in CLASSES:
        (corpus / name).mkdir(parents=True)
    (corpus / CLASSES[0] / "law.txt").write_text("أ ب ج د هـ", encoding="utf-8")
    (corpus / CLASSES[1] / "other.txt").write_text("و ز", encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    encoder = Mock(dimension=2, max_length=32)
    encoder.encode_documents.side_effect = lambda texts: np.array([[1.0, 0.0] for _ in texts])
    factory = Mock(return_value=encoder)
    monkeypatch.setattr(pipeline, "AraBERTEncoder", factory)
    return corpus, model, tmp_path / "output", encoder, factory


def build(setup, **kwargs):
    corpus, model, output, _, _ = setup
    return pipeline.build_artifacts(
        model, corpus, output, chunk_size=2, overlap=0, batch_size=2, **kwargs,
    )


def test_build_persistence_manifest_and_new_snapshot(setup):
    corpus, model, output, encoder, factory = setup
    target = build(setup)
    current = json.loads((output / "current.json").read_text(encoding="utf-8"))
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert current["build_dir"] == str(target)
    assert current["qdrant_path"] == str(target / "qdrant")
    assert manifest["documents"] == 2
    assert manifest["passages"] == 4
    assert manifest["dimension"] == 2
    assert manifest["max_length"] == 32
    assert manifest["truncate"] is False
    assert len(manifest["corpus_sha256"]) == 64
    assert encoder.encode_documents.call_count == 2
    factory.assert_called_once_with(model.resolve(), device="cpu", batch_size=2,
                                    max_length=None, truncate=False)
    with QdrantStore(current["qdrant_path"], dimension=2) as store:
        assert store.count() == 4
        assert store.search([1, 0])[0]["passage"]["metadata"]["filename"]
    (corpus / CLASSES[0] / "law.txt").write_text("نص", encoding="utf-8")
    second = build(setup)
    assert second != target and target.is_dir()
    with QdrantStore(second / "qdrant", dimension=2) as store:
        assert store.count() == 2  # Aucun ancien passage du document modifié.


def test_failed_encoding_preserves_pointer_and_closes_store(setup):
    _, _, output, encoder, _ = setup
    first = build(setup)
    old_pointer = (output / "current.json").read_bytes()
    encoder.encode_documents.side_effect = [np.array([[1., 0.], [1., 0.]]), RuntimeError("failed")]
    with pytest.raises(RuntimeError, match="failed"):
        build(setup)
    assert (output / "current.json").read_bytes() == old_pointer
    incomplete = next(path for path in output.glob("build-*") if path != first)
    assert not (incomplete / "manifest.json").exists()
    with QdrantStore(incomplete / "qdrant", dimension=2) as store:
        assert store.count() == 2


def test_empty_corpus_does_not_load_model(setup):
    corpus, _, output, _, factory = setup
    for path in corpus.rglob("*.txt"):
        path.write_text(" \n", encoding="utf-8")
    with pytest.raises(ValueError, match="Aucun document"):
        build(setup)
    factory.assert_not_called()
    assert not output.exists()


def test_unreadable_strict_and_opt_in_skip(setup):
    corpus, _, _, _, factory = setup
    (corpus / CLASSES[0] / "bad.txt").write_bytes(b"\xff")
    with pytest.raises(UnicodeDecodeError):
        build(setup)
    factory.assert_not_called()
    assert build(setup, strict=False).is_dir()


def test_rejects_output_inside_inputs(setup):
    corpus, model, _, _, factory = setup
    for output in (corpus / "index", model / "index"):
        with pytest.raises(ValueError, match="extérieure"):
            pipeline.build_artifacts(model, corpus, output)
    factory.assert_not_called()


def test_publication_failure_preserves_previous_pointer(setup, monkeypatch):
    _, _, output, _, _ = setup
    build(setup)
    previous = (output / "current.json").read_bytes()
    monkeypatch.setattr(pipeline.os, "replace", Mock(side_effect=OSError("publication failed")))
    with pytest.raises(OSError, match="publication failed"):
        build(setup)
    assert (output / "current.json").read_bytes() == previous
    assert not list(output.glob(".current-*.tmp"))


def test_cli_success_and_failure(monkeypatch, tmp_path, capsys):
    builder = Mock(return_value=tmp_path / "build-test")
    monkeypatch.setattr(pipeline, "build_artifacts", builder)
    assert pipeline.main(["--model-dir", "local", "--truncate", "--skip-unreadable"]) == 0
    assert builder.call_args.args[0] == Path("local")
    assert builder.call_args.kwargs["truncate"] is True
    assert builder.call_args.kwargs["strict"] is False
    builder.side_effect = ValueError("invalid")
    assert pipeline.main(["--model-dir", "local"]) == 1
    assert "invalid" in capsys.readouterr().err

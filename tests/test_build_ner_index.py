"""Pipeline offline NER : corpus réel et annotateur simulé, sans téléchargement."""

import json

import pytest

from scripts import build_ner_index


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    corpus = tmp_path / "raw"
    for name in ("family_inheritance", "real_estate"):
        (corpus / name).mkdir(parents=True)
        (corpus / name / "one.txt").write_text("محمد", encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    calls = []

    class Annotator:
        def __init__(self, path, **kwargs):
            calls.append((path, kwargs))

        def annotate(self, text):
            return [{"text": text, "label": "PER", "start": 0, "end": len(text)}] if text else []

    monkeypatch.setattr(build_ner_index, "AraBERTAnnotator", Annotator)
    return corpus, model, tmp_path / "output" / "index.json", calls


def test_pipeline_loads_model_once_and_saves_annotations_and_index(pipeline):
    corpus, model, output, calls = pipeline
    target = build_ner_index.build_artifacts(model, corpus, output, max_length=128, stride=16)
    result = json.loads(target.read_text(encoding="utf-8"))
    assert target == output.resolve()
    assert len(calls) == 1
    assert calls[0] == (model.resolve(), {
        "device": "cpu", "max_length": 128, "stride": 16, "invalid_bio": "ignore",
    })
    assert result["statistics"]["documents"] == 2
    assert result["statistics"]["entities"] == 2
    assert len(result["entity_index"]["PER"]["محمد"]) == 2
    assert {document["class"] for document in result["documents"]} == {
        "family_inheritance", "real_estate",
    }
    assert all(path.read_text(encoding="utf-8") == "محمد" for path in corpus.rglob("*.txt"))


def test_read_error_is_strict_by_default_and_skip_is_explicit(pipeline):
    corpus, model, output, calls = pipeline
    (corpus / "real_estate" / "invalid.txt").write_bytes(b"\xff")
    with pytest.raises(UnicodeError):
        build_ner_index.build_artifacts(model, corpus, output)
    assert not calls and not output.exists()
    build_ner_index.build_artifacts(model, corpus, output, strict=False)
    assert json.loads(output.read_text(encoding="utf-8"))["statistics"]["documents"] == 2


def test_inference_failure_keeps_previous_export(pipeline, monkeypatch):
    corpus, model, output, _ = pipeline
    output.parent.mkdir()
    output.write_text("previous", encoding="utf-8")

    class BrokenAnnotator:
        def __init__(self, *args, **kwargs):
            pass

        def annotate(self, text):
            raise RuntimeError("inference failed")

    monkeypatch.setattr(build_ner_index, "AraBERTAnnotator", BrokenAnnotator)
    with pytest.raises(RuntimeError, match="inference failed"):
        build_ner_index.build_artifacts(model, corpus, output)
    assert output.read_text(encoding="utf-8") == "previous"


def test_empty_corpus_does_not_replace_export(pipeline):
    corpus, model, output, calls = pipeline
    for path in corpus.rglob("*.txt"):
        path.unlink()
    output.parent.mkdir()
    output.write_text("previous", encoding="utf-8")
    with pytest.raises(ValueError, match="aucun document"):
        build_ner_index.build_artifacts(model, corpus, output)
    assert not calls
    assert output.read_text(encoding="utf-8") == "previous"


@pytest.mark.parametrize("location", ["model", "corpus"])
def test_rejects_output_inside_sources(pipeline, location):
    corpus, model, _, calls = pipeline
    target = (model if location == "model" else corpus) / "config.json"
    target.write_text("source", encoding="utf-8")
    with pytest.raises(ValueError, match="extérieure"):
        build_ner_index.build_artifacts(model, corpus, target)
    assert target.read_text(encoding="utf-8") == "source"
    assert not calls


@pytest.mark.parametrize("options", [
    {"max_length": 2}, {"stride": -1}, {"max_length": 6, "stride": 4},
    {"invalid_bio": "repair"},
])
def test_bad_options_fail_before_loading_model(pipeline, options):
    corpus, model, output, calls = pipeline
    with pytest.raises(ValueError):
        build_ner_index.build_artifacts(model, corpus, output, **options)
    assert not calls and not output.exists()


def test_cli_success_and_failure(pipeline, capsys):
    corpus, model, output, _ = pipeline
    args = ["--model-dir", str(model), "--data-dir", str(corpus), "--output", str(output)]
    assert build_ner_index.main(args) == 0
    assert str(output) in capsys.readouterr().out
    assert build_ner_index.main(args + ["--stride", "-1"]) == 1
    assert "Erreur de construction" in capsys.readouterr().err


def test_cli_forwards_explicit_options(pipeline, monkeypatch):
    corpus, model, output, _ = pipeline
    received = {}

    def build(*args, **kwargs):
        received.update(kwargs)
        return output

    monkeypatch.setattr(build_ner_index, "build_artifacts", build)
    assert build_ner_index.main([
        "--model-dir", str(model), "--data-dir", str(corpus), "--output", str(output),
        "--device", "cuda:0", "--max-length", "256", "--stride", "32",
        "--invalid-bio", "raise", "--skip-unreadable",
    ]) == 0
    assert received == {"device": "cuda:0", "max_length": 256, "stride": 32,
                        "invalid_bio": "raise", "strict": False}


def test_help_exits_before_model_loading(monkeypatch, capsys):
    def no_model(*args, **kwargs):
        raise AssertionError("Le modèle ne doit pas être chargé pour --help")

    monkeypatch.setattr(build_ner_index, "AraBERTAnnotator", no_model)
    with pytest.raises(SystemExit) as exc:
        build_ner_index.main(["--help"])
    assert exc.value.code == 0
    assert "--model-dir" in capsys.readouterr().out

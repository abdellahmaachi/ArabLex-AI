"""Annotation, positions et cohérence des exports, sans modèle externe."""

import json
import re
from copy import deepcopy

import pytest

from src.common.document_loader import load_documents
from src.ner.corpus_indexer import (
    annotate_documents,
    build_entity_index,
    export_annotations,
    index_corpus,
)


class FakeAnnotator:
    def __init__(self):
        self.calls = []

    def annotate(self, text):
        self.calls.append(text)
        return [
            {"text": match.group(), "label": "PER", "start": match.start(), "end": match.end()}
            for match in re.finditer("محمد", text)
        ]


def test_annotation_export_and_index_preserve_repeated_arabic_spans(tmp_path):
    documents = [
        {"id": "b", "text": "حَكَمَ محمد، محمد", "class": "family_inheritance",
         "metadata": {"reviewed": False}},
        {"id": "a", "text": "محمد"}, {"id": "empty", "text": ""},
        {"id": "outside", "text": "نص بلا أسماء"},
    ]
    original = deepcopy(documents)
    annotator = FakeAnnotator()
    output = tmp_path / "nested" / "corpus.json"
    bundle = index_corpus(iter(documents), annotator, output_path=output)
    assert len(annotator.calls) == 4
    assert documents == original
    assert [document["id"] for document in bundle["documents"]] == ["a", "b", "empty", "outside"]
    assert bundle["entity_index"] == {
        "PER": {"محمد": {"a": [{"start": 0, "end": 4}],
                         "b": [{"start": 7, "end": 11}, {"start": 13, "end": 17}]}},
    }
    assert bundle["statistics"] == {
        "documents": 4, "documents_with_entities": 2, "entities": 3,
        "unique_entities": 1, "entities_by_type": {"PER": 3},
    }
    assert json.loads(output.read_text(encoding="utf-8")) == bundle
    assert "محمد" in output.read_text(encoding="utf-8")
    bundle["documents"][1]["metadata"]["reviewed"] = True
    assert documents == original


def test_index_separates_types_and_exact_surfaces():
    documents = [
        {"id": "z", "text": "محمد", "entities": [
            {"text": "محمد", "label": "ORG", "start": 0, "end": 4}]},
        {"id": "a", "text": "مُحمد محمد", "entities": [
            {"text": "محمد", "label": "PER", "start": 6, "end": 10},
            {"text": "مُحمد", "label": "PER", "start": 0, "end": 5}]},
    ]
    before = deepcopy(documents)
    index = build_entity_index(documents)
    assert set(index) == {"PER", "ORG"}
    assert set(index["PER"]) == {"محمد", "مُحمد"}
    assert index == build_entity_index(reversed(documents))
    assert documents == before


def test_duplicate_ids_rejected_before_annotation_or_export(tmp_path):
    annotator = FakeAnnotator()
    output = tmp_path / "existing.json"
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(ValueError, match="dupliqué"):
        index_corpus([{"id": "same", "text": ""}] * 2, annotator, output_path=output)
    assert annotator.calls == []
    assert output.read_text(encoding="utf-8") == "existing"


@pytest.mark.parametrize("entity", [
    {"text": "محمد", "label": "PER", "start": -1, "end": 4},
    {"text": "محمد", "label": "PER", "start": 0, "end": 5},
    {"text": "محمد", "label": "PER", "start": False, "end": 4},
    {"text": "محمد", "label": "PER", "start": 0.0, "end": 4},
    {"text": "آخر", "label": "PER", "start": 0, "end": 4},
    {"text": "محمد", "label": "", "start": 0, "end": 4},
    {"text": "محمد", "label": "PER LOC", "start": 0, "end": 4},
    {"text": "محمد", "start": 0, "end": 4},
])
def test_rejects_invalid_entities(entity):
    with pytest.raises(ValueError):
        build_entity_index([{"id": "a", "text": "محمد", "entities": [entity]}])


def test_rejects_overlap_and_unannotated_documents():
    entity = {"text": "محمد", "label": "PER", "start": 0, "end": 4}
    with pytest.raises(ValueError, match="chevauchantes"):
        build_entity_index([{"id": "a", "text": "محمد", "entities": [entity, entity]}])
    with pytest.raises(TypeError, match="entities"):
        build_entity_index([{"id": "a", "text": "محمد"}])


def test_empty_corpus_and_existing_entities_are_replaced(tmp_path):
    result = index_corpus([], FakeAnnotator(), output_path=tmp_path / "empty.json")
    assert result["documents"] == []
    assert result["entity_index"] == {}
    assert result["statistics"]["entities"] == 0
    assert annotate_documents([{"id": "a", "text": "", "entities": ["stale"]}],
                              FakeAnnotator())[0]["entities"] == []


def test_failed_write_keeps_existing_export_and_cleans_temporary(tmp_path, monkeypatch):
    path = tmp_path / "export.json"
    path.write_text("previous", encoding="utf-8")

    def fail(*args):
        raise OSError("cannot replace")

    monkeypatch.setattr("src.ner.corpus_indexer.os.replace", fail)
    with pytest.raises(OSError, match="cannot replace"):
        export_annotations([], path)
    assert path.read_text(encoding="utf-8") == "previous"
    assert list(tmp_path.iterdir()) == [path]


def test_annotation_failure_does_not_write_partial_output(tmp_path):
    class FailingAnnotator:
        def annotate(self, text):
            raise RuntimeError("inference failed")

    path = tmp_path / "export.json"
    with pytest.raises(RuntimeError, match="inference failed"):
        index_corpus([{"id": "a", "text": "محمد"}], FailingAnnotator(), output_path=path)
    assert not path.exists()


def test_loader_integration_and_source_overwrite_protection(tmp_path):
    root = tmp_path / "raw"
    for name in ("family_inheritance", "real_estate"):
        (root / name).mkdir(parents=True)
        (root / name / "doc.txt").write_text("محمد", encoding="utf-8")
    documents = load_documents(root, strict=True)
    result = index_corpus(documents, FakeAnnotator())
    assert len(result["entity_index"]["PER"]["محمد"]) == 2
    source = root / "family_inheritance" / "doc.txt"
    with pytest.raises(ValueError, match="source"):
        export_annotations(result["documents"], source)
    assert source.read_text(encoding="utf-8") == "محمد"

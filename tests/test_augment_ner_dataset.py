"""Fusion train uniquement et préservation des corpus sources."""

import json

import pytest

from scripts.augment_ner_dataset import augment_dataset
from scripts.generate_synthetic_ner import generate_documents


@pytest.fixture
def corpus(tmp_path):
    rows = generate_documents(2)
    original = tmp_path / "original.jsonl"
    synthetic = tmp_path / "synthetic.jsonl"
    # Pas de saut de ligne final : vérifie la séparation lors de la fusion.
    original.write_text(json.dumps(rows[0], ensure_ascii=False), encoding="utf-8")
    synthetic.write_text(json.dumps(rows[1], ensure_ascii=False) + "\n", encoding="utf-8")
    return original, synthetic, tmp_path / "train_augmented.jsonl"


def test_merge_preserves_records_and_other_splits(corpus):
    original, synthetic, output = corpus
    validation = original.parent / "validation.jsonl"
    test = original.parent / "test.jsonl"
    validation.write_bytes(b"validation unchanged\n")
    test.write_bytes(b"test unchanged\n")
    before = {path: path.read_bytes() for path in (original, synthetic, validation, test)}
    report = augment_dataset(*corpus)
    assert {key: report[key] for key in ("original", "synthetic", "total")} == {
        "original": 1, "synthetic": 1, "total": 2,
    }
    assert report["unique_ids"] == 2
    assert report["bio_view_mismatches"] == []
    assert output.read_bytes() == before[original] + b"\n" + before[synthetic]
    assert all(path.read_bytes() == data for path, data in before.items())


@pytest.mark.parametrize("field,value", [("split", "validation"), ("split", "test"),
                                         ("original_split", "test"), ("split", None)])
@pytest.mark.parametrize("source", [0, 1])
def test_rejects_evaluation_records(corpus, field, value, source):
    path = corpus[source]
    record = json.loads(path.read_text(encoding="utf-8"))
    record[field] = value
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="split"):
        augment_dataset(*corpus)
    assert not corpus[2].exists()


def test_duplicate_ids_fail_without_output(corpus):
    corpus[1].write_bytes(corpus[0].read_bytes())
    with pytest.raises(ValueError, match="dupliqué"):
        augment_dataset(*corpus)
    assert not corpus[2].exists()


@pytest.mark.parametrize("payload", [b"", b"not json\n", b"null\n", b"{}\n"])
def test_invalid_input_does_not_create_output(corpus, payload):
    corpus[1].write_bytes(payload)
    with pytest.raises(ValueError):
        augment_dataset(*corpus)
    assert not corpus[2].exists()


def test_existing_output_and_evaluation_paths_are_protected(corpus):
    original, synthetic, output = corpus
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        augment_dataset(*corpus)
    assert output.read_bytes() == b"existing"
    with pytest.raises(ValueError, match="train_augmented"):
        augment_dataset(original, synthetic, original.parent / "validation.jsonl")


def test_same_input_rejected(corpus):
    with pytest.raises(ValueError, match="distincts"):
        augment_dataset(corpus[0], corpus[0], corpus[2])


@pytest.mark.parametrize("problem", ["offset", "overlap", "text", "label", "bio"])
def test_invalid_annotations_block_export(corpus, problem):
    row = json.loads(corpus[1].read_text(encoding="utf-8"))
    if problem == "offset":
        row["entities"][0]["start"] = True
    elif problem == "overlap":
        row["entities"].insert(1, dict(row["entities"][0]))
    elif problem == "text":
        row["entities"][0]["text"] = "incorrect"
    elif problem == "label":
        row["entities"][0]["label"] = "UNKNOWN"
    else:
        row["tokens"][0]["tag"] = "I-LOC"
    corpus[1].write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError):
        augment_dataset(*corpus)
    assert not corpus[2].exists()


def test_duplicate_text_with_distinct_id_is_rejected(corpus):
    row = json.loads(corpus[0].read_text(encoding="utf-8"))
    row["id"] = "different-id"
    row["text"] = row["text"].replace("رقم 1.", "رقم 2.")
    corpus[1].write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="texte dupliqué"):
        augment_dataset(*corpus)
    assert not corpus[2].exists()


def test_loc_output_and_entity_distribution(corpus):
    from collections import Counter

    output = corpus[2].with_name("train_augmented_loc.jsonl")
    expected = Counter(e["label"] for path in corpus[:2]
                       for e in json.loads(path.read_text(encoding="utf-8"))["entities"])
    report = augment_dataset(corpus[0], corpus[1], output)
    assert report["entities_by_type"] == expected
    assert report["duplicate_ids"] == report["duplicate_texts"] == 0


def test_bio_view_mismatch_is_reported_without_changing_source(corpus):
    row = json.loads(corpus[0].read_text(encoding="utf-8"))
    for token in row["tokens"]:
        token["tag"] = "O"
    corpus[0].write_text(json.dumps(row), encoding="utf-8")
    before = corpus[0].read_bytes()
    report = augment_dataset(*corpus)
    assert report["bio_view_mismatches"] == [row["id"]]
    assert corpus[0].read_bytes() == before
    assert corpus[2].read_bytes().startswith(before)

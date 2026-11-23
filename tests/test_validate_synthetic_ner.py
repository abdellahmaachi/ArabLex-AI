"""Audit des annotations synthétiques et erreurs de corpus."""

import json
from copy import deepcopy

import pytest

from scripts.generate_synthetic_ner import generate_documents
from scripts.validate_synthetic_ner import main, validate_document, validate_files


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
                    encoding="utf-8")
    return path


def test_generated_annotations_are_valid(tmp_path):
    rows = generate_documents(10)
    assert all(validate_document(row) == [] for row in rows)
    result = validate_files([write_jsonl(tmp_path / "train.jsonl", rows)])
    assert result["valid"]
    assert result["documents"] == 10


@pytest.mark.parametrize("field,value,expected", [
    ("start", True, "offsets"), ("end", 999999, "offsets"),
    ("text", "incorrect", "text[start:end]"), ("label", "UNKNOWN", "catégorie inconnue"),
])
def test_invalid_entity_fields(field, value, expected):
    row = generate_documents(1)[0]
    row["entities"][0][field] = value
    assert any(expected in error for error in validate_document(row))


def test_duplicate_entity_and_overlap():
    row = generate_documents(1)[0]
    row["entities"].insert(1, deepcopy(row["entities"][0]))
    errors = validate_document(row)
    assert any("dupliquée" in error for error in errors)
    assert any("chevauchants" in error for error in errors)


def test_bio_mismatch_and_invalid_transition():
    row = generate_documents(1)[0]
    token = next(token for token in row["tokens"] if token["tag"].startswith("B-"))
    original = token["tag"]
    token["tag"] = "I-" + original[2:]
    assert any("BIO invalide" in error for error in validate_document(row))
    for token in row["tokens"]:
        token["tag"] = "O"
    assert any("décodage BIO différent" in error for error in validate_document(row))


def test_uncovered_text_and_categories():
    row = generate_documents(1)[0]
    row["tokens"].pop()
    row["legal_class"] = "unknown"
    entity = row["entities"][0]
    entity["label"] = "LOC" if entity["label"] != "LOC" else "PER"
    errors = validate_document(row)
    assert any("fin du texte" in error for error in errors)
    assert any("legal_class" in error for error in errors)
    assert any("catégorie incohérente" in error for error in errors)


def test_duplicates_across_splits_ignore_serial_number(tmp_path):
    first = generate_documents(1)[0]
    second = deepcopy(first)
    second["split"] = "test"
    # Même longueur : les offsets restent corrects.
    second["text"] = second["text"].replace("رقم 1.", "رقم 2.")
    for token in second["tokens"]:
        token["token"] = second["text"][token["start"]:token["end"]]
    paths = [write_jsonl(tmp_path / "train.jsonl", [first]),
             write_jsonl(tmp_path / "test.jsonl", [second])]
    result = validate_files(paths)
    assert not result["valid"]
    assert any("doublon id" in e["message"] for e in result["errors"])
    assert any("doublon text" in e["message"] for e in result["errors"])


def test_contradictory_labels_across_documents(tmp_path):
    rows = generate_documents(2)
    rows[1]["entities"][0] = {**rows[0]["entities"][0], "label": "LOC"}
    # Le premier span des gabarits est PER ou COURT, jamais LOC.
    result = validate_files([write_jsonl(tmp_path / "train.jsonl", rows)])
    assert any("catégories contradictoires" in e["message"] for e in result["errors"])


@pytest.mark.parametrize("record", [None, [], {}, {"text": "abc", "entities": [None],
                                                         "tokens": [{"start": []}]}])
def test_malformed_records_do_not_crash(record):
    assert validate_document(record)


def test_file_errors_and_cli_exit_codes(tmp_path, capsys):
    valid = write_jsonl(tmp_path / "train.jsonl", generate_documents(1))
    assert main([str(valid)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"]
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text("not json\nnull\n", encoding="utf-8")
    assert main([str(invalid)]) == 1
    errors = json.loads(capsys.readouterr().out)["errors"]
    assert {error["line"] for error in errors} == {1, 2}
    empty = tmp_path / "empty.jsonl"
    empty.touch()
    assert not validate_files([empty, tmp_path / "missing.jsonl"])["valid"]


def test_split_matches_filename(tmp_path):
    result = validate_files([write_jsonl(tmp_path / "test.jsonl", generate_documents(1))])
    assert any("split différent" in e["message"] for e in result["errors"])

"""Offsets synthétiques, compatibilité BIO et export reproductible."""

import json
from collections import Counter

import pytest

from scripts.generate_synthetic_ner import (
    LOC_CITIES,
    generate_documents,
    generate_loc_documents,
    main,
    render_passage,
)
from src.ner.bio_decoder import decode_bio


def test_generated_spans_round_trip_through_bio():
    documents = generate_documents(100)
    assert len({row["id"] for row in documents}) == 100
    assert len({row["text"] for row in documents}) == 100
    for row in documents:
        text, tokens = row["text"], row["tokens"]
        assert {entity["label"] for entity in row["entities"]} == {"PER", "COURT", "ORG", "LOC"}
        assert decode_bio(text, [t["tag"] for t in tokens],
                          [(t["start"], t["end"]) for t in tokens]) == row["entities"]
        end = 0
        for token in tokens:
            assert not text[end:token["start"]].strip()
            assert text[token["start"]:token["end"]] == token["token"]
            end = token["end"]
        assert not text[end:].strip()


def test_repeated_names_diacritics_and_attached_prefix():
    name = "مُحَمَّد  العلوي"
    row = render_passage("⚖️ حضر {person}، و{person}.", {"person": name})
    first, second = row["entities"]
    assert first["start"] == len("⚖️ حضر ")
    assert second["start"] == first["end"] + len("، و")
    assert first["text"] == second["text"] == name
    assert all(row["text"][e["start"]:e["end"]] == name for e in row["entities"])


def test_reproducible_generation():
    assert generate_documents(10, 7) == generate_documents(10, 7)
    assert generate_documents(10, 7) != generate_documents(10, 8)


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_count(count):
    with pytest.raises(ValueError):
        generate_documents(count)


def test_cli_jsonl(tmp_path):
    output = tmp_path / "synthetic" / "validation.jsonl"
    args = ["--output", str(output), "--count", "3", "--seed", "9", "--split", "validation"]
    main(args)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows == generate_documents(3, 9, "validation")
    with pytest.raises(FileExistsError):
        main(args)


def test_targeted_loc_diversity_and_annotations():
    from scripts.validate_synthetic_ner import validate_document

    rows = generate_loc_documents()
    assert len(rows) == len({row["id"] for row in rows}) == 150
    assert len({" ".join(row["text"].split()) for row in rows}) == 150
    assert set(Counter(row["template_id"] for row in rows).values()) == {10}
    locations = Counter()
    for row in rows:
        assert validate_document(row) == []
        assert row["split"] == "train"
        entities = row["entities"]
        locs = [entity["text"] for entity in entities if entity["label"] == "LOC"]
        assert len(locs) == len(set(locs)) == 2
        locations.update(locs)
        assert {e["label"] for e in entities} == {"PER", "COURT", "ORG", "LOC"}
        assert decode_bio(row["text"], [t["tag"] for t in row["tokens"]],
                          [(t["start"], t["end"]) for t in row["tokens"]]) == entities
    assert set(locations) == set(LOC_CITIES)
    assert set(locations.values()) == {10}
    assert rows == generate_loc_documents(seed=42)
    assert rows != generate_loc_documents(seed=43)


def test_targeted_loc_cli_and_train_only(tmp_path):
    output = tmp_path / "train_loc.jsonl"
    main(["--loc-focused", "--output", str(output)])
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows == generate_loc_documents()
    with pytest.raises(SystemExit):
        main(["--loc-focused", "--split", "validation", "--output", str(tmp_path / "bad.jsonl")])
    assert not (tmp_path / "bad.jsonl").exists()


@pytest.mark.parametrize("count", [0, -1, True, 1.5, 451])
def test_targeted_loc_invalid_count(count):
    with pytest.raises(ValueError):
        generate_loc_documents(count)

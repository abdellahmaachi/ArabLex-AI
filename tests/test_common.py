"""Tests des documents, du chargement du corpus et de la normalisation arabe."""

import shutil
from dataclasses import fields, replace
from pathlib import Path

import pytest

from src.common.arabic_normalizer import (
    NormalizationConfig,
    normalize_arabic,
    strip_diacritics,
    strip_tatweel,
)
from src.common.document_loader import CLASSES, load_documents


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "corpus"
    for class_name in ("family_inheritance", "real_estate"):
        (root / class_name).mkdir(parents=True)
    return root


def test_loads_both_classes_and_nested_utf8_files(corpus):
    for class_name in CLASSES:
        (corpus / class_name / "same.txt").write_text("نص قانوني", encoding="utf-8")
    nested = corpus / CLASSES[0] / "nested"
    nested.mkdir()
    (nested / "empty.TXT").write_text("", encoding="utf-8")
    (nested / "ignored.csv").write_text("ignored", encoding="utf-8")

    documents = load_documents(corpus)

    assert len(documents) == 3
    assert len({document["id"] for document in documents}) == 3
    assert [document["class"] for document in documents] == [CLASSES[0], CLASSES[0], CLASSES[1]]
    assert [document["text"] for document in documents] == ["", "نص قانوني", "نص قانوني"]
    assert all(Path(document["path"]).is_absolute() for document in documents)
    assert documents[1]["filename"] == "same.txt"


def test_ids_survive_relocation_additions_and_content_changes(corpus, tmp_path):
    original = corpus / CLASSES[0] / "original.txt"
    original.write_text("before", encoding="utf-8")
    original_id = load_documents(corpus)[0]["id"]
    relocated = tmp_path / "relocated"
    shutil.copytree(corpus, relocated)
    (relocated / CLASSES[0] / "original.txt").write_text("after", encoding="utf-8")
    (relocated / CLASSES[0] / "added.txt").write_text("new", encoding="utf-8")

    documents = load_documents(relocated)

    assert next(doc["id"] for doc in documents if doc["filename"] == "original.txt") == original_id
    assert documents == load_documents(relocated)


def test_invalid_utf8_is_logged_and_skipped_or_raised(corpus, caplog):
    (corpus / CLASSES[0] / "bad.txt").write_bytes(b"\xff")
    (corpus / CLASSES[1] / "good.txt").write_text("valid", encoding="utf-8")

    assert len(load_documents(corpus)) == 1
    assert "bad.txt" in caplog.text
    with pytest.raises(UnicodeDecodeError):
        load_documents(corpus, strict=True)


def test_read_error_is_logged_and_skipped_or_raised(corpus, monkeypatch, caplog):
    (corpus / CLASSES[0] / "blocked.txt").touch()

    def denied(*args, **kwargs):
        raise PermissionError("access denied")

    monkeypatch.setattr(Path, "read_text", denied)
    assert load_documents(corpus) == []
    assert "blocked.txt" in caplog.text
    with pytest.raises(PermissionError):
        load_documents(corpus, strict=True)


def test_missing_class_raises(corpus):
    (corpus / CLASSES[1]).rmdir()
    with pytest.raises(FileNotFoundError, match=CLASSES[1]):
        load_documents(corpus)


def test_invalid_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "missing")
    file = tmp_path / "file.txt"
    file.touch()
    with pytest.raises(NotADirectoryError):
        load_documents(file)


def test_defaults_preserve_original_document():
    original = "إِنَّ الـمَحْكَمَةَ أَقَرَّتْ، ١٢\nعلى مسؤولية."
    document = {"text": original}

    normalized = normalize_arabic(document["text"])

    assert normalized == "ان المحكمة اقرت، ١٢\nعلى مسؤولية."
    assert document == {"text": original}


@pytest.mark.parametrize(
    ("option", "source", "expected"),
    [
        ("normalize_unicode", "ا\u0654", "أ"),
        ("remove_diacritics", "بًٌٍَُِّْٰ", "ب"),
        ("remove_tatweel", "قــانون", "قانون"),
        ("normalize_alef", "أإآٱا", "ااااا"),
        ("normalize_alef_maqsura", "على", "علي"),
        ("normalize_teh_marbuta", "محكمة", "محكمه"),
        ("normalize_hamza", "ؤئء", "ويء"),
        ("normalize_digits", "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789"),
        ("normalize_whitespace", " \tنص\n عربي  ", "نص عربي"),
    ],
)
def test_options_can_be_enabled_independently(option, source, expected):
    disabled = NormalizationConfig(**{field.name: False for field in fields(NormalizationConfig)})
    assert normalize_arabic(source, disabled) == source
    assert normalize_arabic(source, replace(disabled, **{option: True})) == expected


def test_helpers_preserve_other_characters():
    assert strip_diacritics("أَ إِ ء ا\u0654 café!") == "أ إ ء ا\u0654 café!"
    assert strip_tatweel("نـص 123!") == "نص 123!"


@pytest.mark.parametrize("source", ["", "Latin 123, punctuation!", "أَإِآٱ ى ة ؤ ئ ء ـ ١٢"])
def test_normalization_is_idempotent(source):
    for config in (
        NormalizationConfig(),
        NormalizationConfig(**{field.name: True for field in fields(NormalizationConfig)}),
    ):
        normalized = normalize_arabic(source, config)
        assert normalize_arabic(normalized, config) == normalized


def test_default_normalization_handles_decomposed_alef():
    assert normalize_arabic("ا\u0654 ا\u0655 ا\u0653") == "ا ا ا"


def test_rejects_non_text():
    with pytest.raises(TypeError, match="text"):
        normalize_arabic(None)


def test_document_fields_and_class_labels(corpus):
    expected_classes = {"family_inheritance", "real_estate"}
    assert set(CLASSES) == expected_classes
    for class_name in expected_classes:
        (corpus / class_name / "document.txt").write_text("نص", encoding="utf-8")
    extra_class = corpus / "unknown"
    extra_class.mkdir()
    (extra_class / "ignored.txt").write_text("ignored", encoding="utf-8")

    documents = load_documents(corpus)

    assert len(documents) == 2
    assert {document["class"] for document in documents} == expected_classes
    for document in documents:
        assert set(document) == {"id", "class", "filename", "path", "text"}
        assert all(isinstance(value, str) for value in document.values())
        assert document["id"]
        assert Path(document["path"]).parent.name == document["class"]


def test_empty_classes_produce_empty_corpus(corpus):
    assert load_documents(corpus) == []


def test_class_path_must_be_directory(corpus):
    class_path = corpus / "real_estate"
    class_path.rmdir()
    class_path.write_text("not a directory", encoding="utf-8")
    with pytest.raises(NotADirectoryError, match="real_estate"):
        load_documents(corpus)


def test_ids_distinguish_same_filename_in_different_subdirectories(corpus):
    for folder in ("first", "second"):
        directory = corpus / "real_estate" / folder
        directory.mkdir()
        (directory / "same.txt").write_text("same content", encoding="utf-8")

    documents = load_documents(corpus)

    assert len(documents) == 2
    assert documents[0]["id"] != documents[1]["id"]


def test_renaming_document_changes_id(corpus):
    original = corpus / "real_estate" / "before.txt"
    original.write_text("text", encoding="utf-8")
    original_id = load_documents(corpus)[0]["id"]
    original.rename(original.with_name("after.txt"))
    assert load_documents(corpus)[0]["id"] != original_id


def test_normalization_preserves_loaded_document_and_source_file(corpus):
    original = "إِنَّ الـمَحْكَمَةَ"
    source = corpus / "family_inheritance" / "original.txt"
    source.write_text(original, encoding="utf-8")
    original_bytes = source.read_bytes()
    document = load_documents(corpus)[0]
    before = document.copy()

    normalized = normalize_arabic(document["text"])

    assert normalized == "ان المحكمة"
    assert document == before
    assert document["text"] == original
    assert source.read_bytes() == original_bytes

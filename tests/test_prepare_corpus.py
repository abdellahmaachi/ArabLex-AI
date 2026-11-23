"""Vérifie les sorties du script exécuté depuis un autre répertoire."""

import subprocess
import sys
from pathlib import Path

import pytest

from src.common.document_loader import CLASSES

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_corpus.py"


@pytest.fixture
def corpus(tmp_path):
    for class_name in CLASSES:
        (tmp_path / class_name).mkdir()
    return tmp_path


def run_script(root):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--data-dir", str(root)],
        cwd=root,
        capture_output=True,
        check=False,
    )


def test_reports_counts_including_empty_class(corpus):
    (corpus / CLASSES[0] / "one.txt").write_text("نص قانوني", encoding="utf-8")
    (corpus / CLASSES[0] / "two.txt").write_text("", encoding="utf-8")

    result = run_script(corpus)

    assert result.returncode == 0
    assert b"family_inheritance : 2" in result.stdout
    assert b"real_estate : 0" in result.stdout
    assert b"Total : 2" in result.stdout
    assert result.stderr == b""


def test_reports_bad_file_and_counts_only_loaded_documents(corpus):
    (corpus / CLASSES[0] / "bad.txt").write_bytes(b"\xff")
    (corpus / CLASSES[1] / "good.txt").write_text("valid", encoding="utf-8")

    result = run_script(corpus)

    assert result.returncode == 0
    assert b"family_inheritance : 0" in result.stdout
    assert b"real_estate : 1" in result.stdout
    assert b"Total : 1" in result.stdout
    assert b"bad.txt" in result.stderr
    assert b"WARNING" in result.stderr


def test_reports_missing_class_without_traceback(corpus):
    (corpus / CLASSES[1]).rmdir()

    result = run_script(corpus)

    assert result.returncode == 1
    assert b"Erreur de chargement" in result.stderr
    assert b"real_estate" in result.stderr
    assert b"Traceback" not in result.stderr
    assert result.stdout == b""

"""Fusionne les JSONL d'entraînement, sans ouvrir Validation ni Test.

Usage : python scripts/augment_ner_dataset.py
Ajout LOC : python scripts/augment_ner_dataset.py --loc-focused
Conserve les lignes originales puis synthétiques, leurs annotations et offsets.
Offsets, chevauchements, BIO, catégories et doublons sont vérifiés avant export.
Les écarts historiques BIO/spans sont rapportés sans modifier les annotations.
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ner.bio_decoder import decode_bio


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "annotations" / "ner"
DEFAULT_ORIGINAL = DATA_DIR / "silver" / "train.jsonl"
DEFAULT_SYNTHETIC = DATA_DIR / "synthetic" / "train.jsonl"
DEFAULT_OUTPUT = DATA_DIR / "augmented" / "train_augmented.jsonl"
LOC_SYNTHETIC = DATA_DIR / "synthetic" / "train_loc.jsonl"
LOC_OUTPUT = DATA_DIR / "augmented" / "train_augmented_loc.jsonl"
ENTITY_TYPES = ("PER", "COURT", "ORG", "LOC")
BIO_LABELS = {"O"} | {f"{prefix}-{kind}" for prefix in ("B", "I") for kind in ENTITY_TYPES}


def _check_annotations(record: dict, location: str) -> bool:
    """Valide les deux vues ; retourne si BIO et spans diffèrent."""
    text = record["text"]
    for field, text_key in (("entities", "text"), ("tokens", "token")):
        previous_end = 0
        for index, item in enumerate(record[field]):
            where = f"{location} : {field}[{index}]"
            if not isinstance(item, dict):
                raise ValueError(f"{where} : objet attendu")
            start, end = item.get("start"), item.get("end")
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
                raise ValueError(f"{where} : offsets invalides")
            if start < previous_end:
                raise ValueError(f"{where} : chevauchement, doublon ou ordre invalide")
            if item.get(text_key) != text[start:end]:
                raise ValueError(f"{where} : texte différent des offsets")
            label = item.get("label" if field == "entities" else "tag")
            allowed = ENTITY_TYPES if field == "entities" else BIO_LABELS
            if not isinstance(label, str) or label not in allowed:
                raise ValueError(f"{where} : catégorie invalide")
            if field == "tokens" and text[previous_end:start].strip():
                raise ValueError(f"{where} : texte non couvert")
            previous_end = end
        if field == "tokens" and text[previous_end:].strip():
            raise ValueError(f"{location} : fin du texte non couverte")
    try:
        decoded = decode_bio(text, [t["tag"] for t in record["tokens"]],
                             [(t["start"], t["end"]) for t in record["tokens"]])
    except ValueError as exc:
        raise ValueError(f"{location} : {exc}") from exc
    entities = [{key: e[key] for key in ("text", "label", "start", "end")}
                for e in record["entities"]]
    return decoded != entities


def _text_key(text: str) -> str:
    # Compare les contenus, sans que le numéro artificiel masque un doublon.
    body = re.sub(r"^نص تدريبي اصطناعي — ملف رقم \d+\.\s*", "", text)
    return hashlib.sha256(" ".join(body.split()).encode("utf-8")).hexdigest()


def augment_dataset(
    original: Path = DEFAULT_ORIGINAL,
    synthetic: Path = DEFAULT_SYNTHETIC,
    output: Path = DEFAULT_OUTPUT,
) -> dict:
    """Valide les deux entrées puis écrit un nouveau fichier, sans écrasement.

    L'ordre et tous les champs sont conservés, sans rééquilibrage, renommage
    d'identifiants, normalisation de texte ni réassignation de split.
    """
    original, synthetic, output = (Path(path).resolve() for path in (original, synthetic, output))
    if output.name not in {"train_augmented.jsonl", "train_augmented_loc.jsonl"}:
        raise ValueError("La sortie doit se nommer train_augmented.jsonl ou train_augmented_loc.jsonl")
    if original == synthetic or original.samefile(synthetic):
        raise ValueError("Les deux corpus d'entrée doivent être distincts")
    if output in (original, synthetic):
        raise ValueError("La sortie doit être distincte des entrées")
    if output.exists():
        raise FileExistsError(f"La sortie existe déjà : {output}")
    seen_ids: dict[str, str] = {}
    seen_texts: dict[str, str] = {}
    lines: list[bytes] = []
    counts, distributions, source_hashes = {}, {}, {}
    mismatches = []
    for kind, path in (("original", original), ("synthetic", synthetic)):
        if path.stem.lower() in {"validation", "test", "dev", "val"}:
            raise ValueError(f"Corpus d'évaluation interdit : {path}")
        count = 0
        entity_counts = Counter()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for number, raw in enumerate(stream, 1):
                digest.update(raw)
                location = f"{path}:{number}"
                try:
                    record = json.loads(raw.decode("utf-8"))
                except (ValueError, UnicodeError) as exc:
                    raise ValueError(f"{location} : JSON UTF-8 invalide") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{location} : objet JSON attendu")
                if record.get("split") != "train":
                    raise ValueError(f"{location} : seul split='train' est autorisé")
                if "original_split" in record and record["original_split"] != "train":
                    raise ValueError(f"{location} : original_split doit être train")
                for field in ("id", "text", "legal_class"):
                    if not isinstance(record.get(field), str) or not record[field].strip():
                        raise ValueError(f"{location} : {field} doit être une chaîne non vide")
                for field in ("entities", "tokens"):
                    if not isinstance(record.get(field), list):
                        raise ValueError(f"{location} : {field} doit être une liste")
                if kind == "synthetic" and record.get("source") != "synthetic":
                    raise ValueError(f"{location} : source synthétique requise")
                identifier = record["id"]
                if identifier in seen_ids:
                    raise ValueError(f"{location} : id dupliqué, déjà présent dans {seen_ids[identifier]}")
                seen_ids[identifier] = location
                fingerprint = _text_key(record["text"])
                if fingerprint in seen_texts:
                    raise ValueError(f"{location} : texte dupliqué, déjà présent dans {seen_texts[fingerprint]}")
                seen_texts[fingerprint] = location
                if _check_annotations(record, location):
                    mismatches.append(identifier)
                entity_counts.update(entity["label"] for entity in record["entities"])
                lines.append(raw if raw.endswith(b"\n") else raw + b"\n")
                count += 1
        if count == 0:
            raise ValueError(f"Corpus vide : {path}")
        counts[kind] = count
        distributions[kind] = {label: entity_counts[label] for label in ENTITY_TYPES}
        source_hashes[str(path)] = digest.hexdigest()
    # Aucune ouverture en écriture avant validation complète des deux entrées.
    for path, expected in source_hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source modifiée pendant la lecture : {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        try:
            stream.writelines(lines)
        except OSError:
            stream.close()
            output.unlink()
            raise
    return {
        **counts, "total": sum(counts.values()), "unique_ids": len(seen_ids),
        "entities_by_source": distributions,
        "entities_by_type": {label: sum(row[label] for row in distributions.values())
                             for label in ENTITY_TYPES},
        "bio_view_mismatches": mismatches, "source_sha256": source_hashes,
        "duplicate_ids": 0, "duplicate_texts": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--synthetic", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--loc-focused", action="store_true",
                        help="Fusionner Train augmenté et train_loc.jsonl")
    args = parser.parse_args(argv)
    args.original = args.original or (DEFAULT_OUTPUT if args.loc_focused else DEFAULT_ORIGINAL)
    args.synthetic = args.synthetic or (LOC_SYNTHETIC if args.loc_focused else DEFAULT_SYNTHETIC)
    args.output = args.output or (LOC_OUTPUT if args.loc_focused else DEFAULT_OUTPUT)
    try:
        counts = augment_dataset(args.original, args.synthetic, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Erreur : {exc}\n")
    print(json.dumps({"output": str(args.output), **counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

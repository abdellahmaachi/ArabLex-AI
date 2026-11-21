"""Construit un index RAG Qdrant Local avec un checkpoint AraBERT local.

Usage : python scripts/build_rag_index.py --model-dir artifacts/rag/encoder
Chaque exécution crée un dossier build-<uuid> dans artifacts/rag. current.json
est remplacé atomiquement après fermeture du stockage et écriture du manifeste.
Les anciennes constructions sont conservées. Aucun appel Gemini, entraînement
ou téléchargement n'est effectué. Un échec peut laisser un dossier incomplet,
jamais publié comme index courant.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.document_loader import DEFAULT_DATA_DIR
from src.rag.arabert_encoder import AraBERTEncoder
from src.rag.chunker import chunk_documents
from src.rag.ingestion import DEFAULT_NORMALIZATION, load_rag_documents
from src.rag.qdrant_store import QdrantStore

LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "rag"


def _publish(path: Path, payload: dict) -> None:
    """Publie le pointeur dans le même système de fichiers par os.replace."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".current-",
            suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def build_artifacts(
    model_dir: str | Path,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    collection_name: str = "legal_passages",
    chunk_size: int = 200,
    overlap: int = 40,
    batch_size: int = 32,
    device: str = "cpu",
    max_length: int | None = None,
    truncate: bool = False,
    strict: bool = True,
) -> Path:
    """Construit et publie un index complet ; renvoie le dossier build créé.

    Le manifeste contient chemins, dimension, effectifs et paramètres nécessaires
    à la recherche. ``current.json`` indique le chemin Qdrant, la collection et
    le manifeste à utiliser. Chaque construction repart du corpus actuel : aucun
    ancien passage supprimé ou modifié ne subsiste dans le nouvel index.

    Les textes préparés et passages résident en mémoire ; les embeddings sont
    calculés et écrits par lots. Le modèle est chargé une fois. Le corpus vide est
    refusé et les fichiers illisibles sont bloquants sauf avec strict=False.
    Par défaut, un passage dépassant la limite en sous-tokens provoque une erreur :
    réduire chunk_size ou activer explicitement truncate pour autoriser la perte.
    """
    chunk_documents([], chunk_size=chunk_size, overlap=overlap)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size doit être un entier strictement positif")
    if max_length is not None and (type(max_length) is not int or max_length < 3):
        raise ValueError("max_length doit être un entier supérieur ou égal à 3")
    if type(truncate) is not bool or type(strict) is not bool:
        raise TypeError("truncate et strict doivent être des booléens")
    if not isinstance(collection_name, str) or not collection_name.strip():
        raise ValueError("collection_name doit être une chaîne non vide")
    model_path = Path(model_dir).resolve(strict=True)
    corpus_path = Path(data_dir).resolve(strict=True)
    output = Path(output_dir).resolve()
    for name, path in (("modèle", model_path), ("corpus", corpus_path)):
        if not path.is_dir():
            raise NotADirectoryError(f"Le {name} n'est pas un dossier : {path}")
        if output.is_relative_to(path):
            raise ValueError(f"La sortie doit être extérieure au dossier du {name}")
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"La sortie n'est pas un dossier : {output}")
    LOGGER.info("Chargement et préparation du corpus : %s", corpus_path)
    documents = load_rag_documents(corpus_path, strict=strict)
    if not documents:
        raise ValueError("Aucun document exploitable ; indexation annulée")
    passages = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
    if not passages:
        raise ValueError("Aucun passage exploitable ; indexation annulée")
    LOGGER.info("Chargement du modèle local : %s", model_path)
    encoder = AraBERTEncoder(
        model_path, device=device, batch_size=batch_size, max_length=max_length, truncate=truncate,
    )
    target = output / f"build-{uuid4().hex}"
    target.mkdir(parents=True, exist_ok=False)
    qdrant_path = target / "qdrant"
    with QdrantStore(qdrant_path, dimension=encoder.dimension, collection_name=collection_name) as store:
        for start in range(0, len(passages), batch_size):
            batch = passages[start:start + batch_size]
            vectors = encoder.encode_documents(passage["text"] for passage in batch)
            store.upsert(batch, vectors, batch_size=batch_size)
            LOGGER.info("Passages indexés : %d/%d", start + len(batch), len(passages))
        if store.count() != len(passages):
            raise RuntimeError("Nombre de points Qdrant différent du nombre de passages")
    fingerprint = hashlib.sha256()
    for document in documents:
        fingerprint.update(json.dumps(
            [document["id"], document["text"], document["prepared_text"]],
            ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8"))
        fingerprint.update(b"\n")
    manifest = {
        "schema_version": 1, "data_dir": str(corpus_path), "model_dir": str(model_path),
        "qdrant_path": str(qdrant_path), "collection_name": collection_name,
        "dimension": encoder.dimension, "distance": "Cosine",
        "documents": len(documents), "passages": len(passages),
        "documents_by_class": dict(Counter(document["class"] for document in documents)),
        "corpus_sha256": fingerprint.hexdigest(),
        "normalization": asdict(DEFAULT_NORMALIZATION),
        "chunk_size": chunk_size, "overlap": overlap, "chunk_unit": "whitespace_word",
        "batch_size": batch_size, "device": device, "max_length": encoder.max_length,
        "truncate": truncate, "strict": strict,
    }
    manifest_path = target / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    _publish(output / "current.json", {
        "schema_version": 1, "build_dir": str(target), "manifest": str(manifest_path),
        "qdrant_path": str(qdrant_path), "collection_name": collection_name,
        "dimension": encoder.dimension,
    })
    LOGGER.info("Index publié : %d documents, %d passages", len(documents), len(passages))
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="Checkpoint AraBERT local")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--collection-name", default="legal_passages")
    parser.add_argument("--chunk-size", type=int, default=200, help="Taille en mots")
    parser.add_argument("--overlap", type=int, default=40, help="Chevauchement en mots")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", type=int, default=None, help="Limite en sous-tokens")
    parser.add_argument("--truncate", action="store_true", help="Autoriser la troncature par le tokenizer")
    parser.add_argument("--skip-unreadable", action="store_true", help="Autoriser un corpus partiel")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")
    try:
        target = build_artifacts(
            args.model_dir, args.data_dir, args.output_dir,
            collection_name=args.collection_name, chunk_size=args.chunk_size,
            overlap=args.overlap, batch_size=args.batch_size, device=args.device,
            max_length=args.max_length, truncate=args.truncate, strict=not args.skip_unreadable,
        )
    except (OSError, UnicodeError, ValueError, TypeError, ImportError, RuntimeError) as error:
        print(f"Erreur de construction de l'index RAG : {error}", file=sys.stderr)
        return 1
    print(f"Index RAG sauvegardé : {target}")
    print(f"Pointeur de l'index courant : {target.parent / 'current.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

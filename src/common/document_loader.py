"""Chargement du corpus juridique organisé en dossiers de classes."""

import logging
import os
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from src.common.document_schema import Document

LOGGER = logging.getLogger(__name__)
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
CLASSES = ("family_inheritance", "real_estate")


def _raise_walk_error(error: OSError) -> None:
    raise error


def load_documents(
    data_dir: str | Path = DEFAULT_DATA_DIR, *, strict: bool = False
) -> list[Document]:
    """Charge récursivement les fichiers .txt UTF-8 des deux classes.

    Les documents sont triés par classe puis par chemin relatif. Leur identifiant
    UUID5 dépend uniquement du chemin relatif au corpus (classe incluse), pas du
    contenu ni de l'emplacement du corpus. Renommer un fichier change son ID.
    Le champ ``path`` contient le chemin absolu et ``text`` le texte non normalisé.

    Un fichier illisible ou non UTF-8 est ignoré avec un avertissement, sauf en
    mode ``strict``, où son exception est propagée. Un dossier de classe absent,
    invalide ou impossible à parcourir provoque toujours une exception afin de
    ne pas renvoyer silencieusement un corpus incomplet. Les fichiers vides sont
    conservés. Les liens symboliques vers des fichiers ou dossiers sont ignorés.
    """
    root = Path(data_dir).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"Le corpus n'est pas un dossier : {root}")

    # Valider les deux classes avant de commencer le chargement.
    for class_name in CLASSES:
        class_dir = root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Dossier de classe absent : {class_dir}")
        if not class_dir.is_dir() or class_dir.is_symlink():
            raise NotADirectoryError(f"Dossier de classe invalide : {class_dir}")

    documents: list[Document] = []
    for class_name in CLASSES:
        paths: list[Path] = []
        for directory, _, filenames in os.walk(
            root / class_name, onerror=_raise_walk_error, followlinks=False
        ):
            for filename in filenames:
                path = Path(directory) / filename
                if path.suffix.lower() == ".txt" and not path.is_symlink():
                    paths.append(path)

        for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as error:
                if strict:
                    raise
                LOGGER.warning("Fichier ignoré %s : %s", path, error)
                continue

            relative_path = path.relative_to(root).as_posix()
            document: Document = {
                "id": str(uuid5(NAMESPACE_URL, f"arabic-legal-text-mining/{relative_path}")),
                "class": class_name,
                "filename": path.name,
                "path": str(path),
                "text": text,
            }
            documents.append(document)

    return documents

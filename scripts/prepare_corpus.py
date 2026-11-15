"""Affiche les effectifs du corpus et les erreurs de chargement.

Usage : python scripts/prepare_corpus.py [--data-dir CHEMIN]
Les fichiers illisibles sont signalés sur stderr et ignorés. Une erreur bloquante
(par exemple un dossier de classe absent) entraîne un code de sortie égal à 1.
Ce script ne modifie ni les fichiers ni leur texte.
"""

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

# Permet aussi l'exécution directe sans installation préalable du projet.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.document_loader import CLASSES, DEFAULT_DATA_DIR, load_documents


def main(argv: list[str] | None = None) -> int:
    """Charge le corpus et affiche le nombre de fichiers effectivement lus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Racine contenant les deux dossiers de classes (par défaut : data/raw).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s : %(message)s")

    try:
        documents = load_documents(args.data_dir)
    except (OSError, UnicodeError) as error:
        print(f"Erreur de chargement : {error}", file=sys.stderr)
        return 1

    counts = Counter(document["class"] for document in documents)
    print("Documents chargés par classe :")
    for class_name in CLASSES:
        print(f"  {class_name} : {counts[class_name]}")
    print(f"Total : {len(documents)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

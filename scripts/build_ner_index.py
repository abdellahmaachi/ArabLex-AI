"""Annote le corpus et sauvegarde son index NER hors ligne.

Exemple depuis la racine du dépôt :
    python scripts/build_ner_index.py --model-dir artifacts/ner/mon_run/best_model

Le modèle et le tokenizer doivent être disponibles localement. Aucun entraînement
ni téléchargement n'est effectué. Le JSON contient les documents annotés, l'index
inversé et les statistiques. Un export existant est remplacé atomiquement après
réussite de toutes les annotations. Les données et le modèle restent inchangés.
"""

import argparse
import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.document_loader import DEFAULT_DATA_DIR, load_documents
from src.ner.annotator import AraBERTAnnotator
from src.ner.corpus_indexer import index_corpus

LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / "artifacts" / "ner" / "entity_index" / "index.json"
)


def build_artifacts(
    model_dir: str | Path,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    device: str = "cpu",
    max_length: int | None = None,
    stride: int | None = None,
    invalid_bio: str = "ignore",
    strict: bool = True,
) -> Path:
    """Charge un seul annotateur, traite tout le corpus et renvoie le JSON écrit.

    Les erreurs de lecture sont bloquantes par défaut. ``strict=False`` autorise
    le chargeur à ignorer les fichiers illisibles en les journalisant. Les erreurs
    d'inférence ou d'annotation sont toujours bloquantes. Un corpus sans document
    est refusé pour ne pas remplacer un index existant par un résultat vide.
    """
    model_path = Path(model_dir).resolve(strict=True)
    if not model_path.is_dir():
        raise NotADirectoryError(f"Le modèle n'est pas un dossier : {model_path}")
    corpus_path = Path(data_dir).resolve(strict=True)
    if not corpus_path.is_dir():
        raise NotADirectoryError(f"Le corpus n'est pas un dossier : {corpus_path}")
    target = Path(output_path).resolve()
    if target.suffix.lower() != ".json":
        raise ValueError("La sortie doit être un fichier .json")
    if target.is_relative_to(model_path) or target.is_relative_to(corpus_path):
        raise ValueError("La sortie doit être extérieure aux dossiers du modèle et du corpus")
    if target.is_dir():
        raise IsADirectoryError(f"La sortie est un dossier : {target}")
    if max_length is not None and (type(max_length) is not int or max_length < 3):
        raise ValueError("max_length doit être un entier supérieur ou égal à 3")
    if stride is not None and (type(stride) is not int or stride < 0):
        raise ValueError("stride doit être un entier positif ou nul")
    if max_length is not None and stride is not None and stride >= max_length - 2:
        raise ValueError("stride doit être inférieur à max_length - 2")
    if invalid_bio not in {"ignore", "raise"}:
        raise ValueError("invalid_bio doit être 'ignore' ou 'raise'")

    LOGGER.info("Chargement du corpus : %s", corpus_path)
    documents = load_documents(corpus_path, strict=strict)
    if not documents:
        raise ValueError("Le corpus ne contient aucun document ; export annulé")
    LOGGER.info("Chargement du modèle local : %s", model_path)
    annotator = AraBERTAnnotator(
        model_path, device=device, max_length=max_length, stride=stride, invalid_bio=invalid_bio,
    )
    LOGGER.info("Annotation de %d documents sur %s", len(documents), device)
    result = index_corpus(documents, annotator, output_path=target)
    statistics = result["statistics"]
    LOGGER.info(
        "Terminé : %d documents, %d entités, %d couples (type, texte) distincts",
        statistics["documents"], statistics["entities"], statistics["unique_entities"],
    )
    for label, count in statistics["entities_by_type"].items():
        LOGGER.info("%s : %d occurrences", label, count)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Dossier local best_model contenant modèle et tokenizer")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Corpus avec les dossiers family_inheritance et real_estate")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH,
                        help="Fichier JSON de sortie (remplacement atomique)")
    parser.add_argument("--device", default="cpu", help="Périphérique PyTorch : cpu, cuda, cuda:0…")
    parser.add_argument("--max-length", type=int, default=None,
                        help="Longueur des fenêtres, tokens spéciaux inclus (défaut : export du modèle)")
    parser.add_argument("--stride", type=int, default=None,
                        help="Chevauchement des fenêtres (défaut : annotateur)")
    parser.add_argument("--invalid-bio", choices=("ignore", "raise"), default="ignore",
                        help="Ignorer les I isolés (IOB2 strict) ou arrêter avec une erreur")
    parser.add_argument("--skip-unreadable", action="store_true",
                        help="Autoriser un corpus partiel en ignorant les fichiers illisibles")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")
    try:
        target = build_artifacts(
            args.model_dir, args.data_dir, args.output,
            device=args.device, max_length=args.max_length, stride=args.stride,
            invalid_bio=args.invalid_bio, strict=not args.skip_unreadable,
        )
    except (OSError, UnicodeError, ValueError, TypeError, ImportError, RuntimeError) as error:
        print(f"Erreur de construction de l'index NER : {error}", file=sys.stderr)
        return 1
    print(f"Documents annotés et index des entités sauvegardés : {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

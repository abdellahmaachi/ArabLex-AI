"""Construit les artefacts offline et publie leur emplacement dans current.json.

Usage : python scripts/build_lexical_index.py --data-dir data/raw
Chaque exécution crée un dossier build-<identifiant> dans artifacts/lexical_search.
Les anciennes constructions sont conservées. Les fichiers illisibles sont signalés
et ignorés, sauf avec --strict. Aucune donnée externe n'est téléchargée.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from uuid import uuid4

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.document_loader import DEFAULT_DATA_DIR, load_documents
from src.lexical_search.preprocessing import PreprocessingConfig
from src.lexical_search.service import LexicalSearchService

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "lexical_search"


def build_artifacts(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    config: PreprocessingConfig | None = None,
    chi2_k: int = 100,
    chi2_min_score: float = 0.0,
    strict: bool = False,
) -> Path:
    """Construit, sauvegarde et publie un index complet ; renvoie son dossier.

    ``current.json`` est remplacé atomiquement seulement après la sauvegarde de
    tous les fichiers. Une erreur conserve donc le précédent pointeur actif.
    Un dossier de construction incomplet peut subsister en cas d'échec d'écriture.
    """
    service = LexicalSearchService(config, chi2_k=chi2_k, chi2_min_score=chi2_min_score)
    documents = load_documents(data_dir, strict=strict)
    service.fit(documents)
    root = Path(output_dir).resolve()
    build_id = f"build-{uuid4().hex}"
    target = root / build_id
    service.save_artifacts(target)
    # Journaliser aussi la provenance et le mode de traitement des erreurs.
    (target / "build.json").write_text(
        json.dumps(
            {"data_dir": str(Path(data_dir).resolve()), "strict": strict},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    pending = root / f".{build_id}.json.tmp"
    try:
        pending.write_text(
            json.dumps({"version": 1, "directory": build_id}, indent=2), encoding="utf-8"
        )
        os.replace(pending, root / "current.json")
    finally:
        pending.unlink(missing_ok=True)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chi2-k", type=int, default=100, help="Nombre de termes par classe")
    parser.add_argument("--chi2-min-score", type=float, default=0.0)
    parser.add_argument("--no-stemming", action="store_true", help="Désactiver la racinisation")
    parser.add_argument("--strict", action="store_true", help="Arrêter sur un fichier illisible")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s : %(message)s")
    try:
        target = build_artifacts(
            args.data_dir, args.output_dir,
            config=PreprocessingConfig(stemming=not args.no_stemming),
            chi2_k=args.chi2_k, chi2_min_score=args.chi2_min_score, strict=args.strict,
        )
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        print(f"Erreur de construction de l'index lexical : {error}", file=sys.stderr)
        return 1
    for label, count in manifest["class_counts"].items():
        print(f"{label} : {count} documents")
    print(f"Total : {manifest['n_documents']} documents, {manifest['matrix_shape'][1]} termes")
    print(f"Artefacts sauvegardés : {target}")
    print(f"Index actif : {target.parent / 'current.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

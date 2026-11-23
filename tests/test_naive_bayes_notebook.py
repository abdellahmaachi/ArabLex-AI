"""Structure du notebook et exécution complète sur un petit corpus UTF-8."""

import json
from pathlib import Path

from src.classification.predictor import NaiveBayesPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "01_naive_bayes.ipynb"


def test_notebook_pipeline_on_synthetic_corpus(tmp_path, monkeypatch):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert len({cell["id"] for cell in notebook["cells"]}) == len(notebook["cells"])
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell["execution_count"] is None and cell["outputs"] == [] for cell in code_cells)
    for index, cell in enumerate(code_cells):
        compile("".join(cell["source"]), f"notebook_cell_{index}", "exec")

    data_dir = tmp_path / "raw"
    for label, text in [
        ("real_estate", "عقد ملكية العقار"),
        ("family_inheritance", "إرث ميراث تركة"),
    ]:
        folder = data_dir / label
        folder.mkdir(parents=True)
        for index in range(8):
            (folder / f"{index}.txt").write_text(text, encoding="utf-8")
    monkeypatch.chdir(PROJECT_ROOT)
    namespace = {"__name__": "__main__"}
    for cell in code_cells:
        source = "".join(cell["source"])
        exec(compile(source, f"notebook_{cell['id']}", "exec"), namespace)
        if cell["id"] == "nb-02":
            namespace["display"] = lambda *args, **kwargs: None
        if cell["id"] == "nb-04":
            namespace.update(
                DATA_DIR=data_dir, OUTPUT_DIR=tmp_path / "artifacts",
                N_SPLITS=2, ALPHAS=(0.5, 1.0),
            )

    output = namespace["run_dir"]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_documents"] == 16
    assert manifest["training_scope"] == "train_partition_only"
    assert all((output / name).is_file() for name in manifest["files"])
    split = json.loads((output / "split.json").read_text(encoding="utf-8"))
    assert not set(split["train_document_ids"]) & set(split["test_document_ids"])
    assert len(split["train_document_ids"]) + len(split["test_document_ids"]) == 16
    predictor = NaiveBayesPredictor.load(output / "predictor.joblib")
    assert predictor.predict_many(namespace["test_texts"]) == namespace["test_predictions"].tolist()

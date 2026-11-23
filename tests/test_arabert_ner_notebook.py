"""Contrôles de l'audit du notebook sur des annotations synthétiques."""

import hashlib
import inspect
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from statistics import mean, median
from types import SimpleNamespace

import pytest

from src.ner.bio_decoder import decode_bio

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "02_arabert_ner.ipynb"


@pytest.fixture
def notebook():
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    namespace = {
        "hashlib": hashlib, "json": json, "decode_bio": decode_bio,
        "Counter": Counter, "mean": mean, "median": median,
    }
    for cell in cells:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            compile(source, f"{NOTEBOOK}:{cell['id']}", "exec")
            if cell["id"] in {"load-functions", "audit-functions", "alignment-functions"}:
                exec(source, namespace)
    return namespace, {cell["id"]: "".join(cell["source"]) for cell in cells}


@pytest.fixture
def document():
    return {
        "id": "example", "split": "train", "legal_class": "family_inheritance",
        "text": "محمد.",
        "tokens": [{"token": "محمد.", "start": 0, "end": 5, "tag": "B-PER"}],
        "entities": [{"text": "محمد.", "start": 0, "end": 5, "label": "PER"}],
    }


def test_valid_annotations_and_exact_boundary_mismatch(notebook, document):
    functions, _ = notebook
    assert functions["check_document"](document, "train", ["O", "B-PER", "I-PER"])
    document["entities"][0].update(end=4, text="محمد")
    with pytest.raises(ValueError, match="décodage BIO exact"):
        functions["check_document"](document, "train", ["O", "B-PER", "I-PER"])


@pytest.mark.parametrize("problem", ["tag", "transition", "offset", "text", "hash", "split"])
def test_corrupted_annotations(notebook, document, problem):
    if problem == "tag":
        document["tokens"][0]["tag"] = "B-UNKNOWN"
    elif problem == "transition":
        document["tokens"][0]["tag"] = "I-PER"
    elif problem == "offset":
        document["tokens"][0]["end"] = 99
    elif problem == "text":
        document["tokens"][0]["token"] = "other"
    elif problem == "hash":
        document["sha256"] = "incorrect"
    else:
        document["split"] = "test"
    with pytest.raises(ValueError):
        notebook[0]["check_document"](document, "train", ["O", "B-PER", "I-PER"])


def test_cross_split_duplicates(notebook, document):
    other = deepcopy(document)
    other["split"] = "test"
    _, errors = notebook[0]["audit_datasets"](
        {"train": [document], "test": [other]}, ["O", "B-PER", "I-PER"]
    )
    assert len(errors) == 2
    assert all("doublon" in error["error"] for error in errors)


def test_stats_include_inconsistent_annotations(notebook, document, capsys):
    functions, cells = notebook
    document["entities"][0].update(end=4, text="محمد")
    functions.update(datasets={"train": [document]}, labels=["O", "B-PER", "I-PER"])
    exec(cells["audit-data"], functions)
    exec(cells["statistics"], functions)
    assert functions["summary"][0]["valides"] == 0
    assert functions["summary"][0]["entités"] == 1
    exec(cells["final-check"], functions)
    assert functions["annotation_errors"][0]["kind"] == "token_view"
    functions["annotation_errors"].append({"kind": "invalid", "error": "invalid offsets"})
    with pytest.raises(ValueError, match="bloquante"):
        exec(cells["final-check"], functions)


def test_jsonl_reports_line_and_rejects_empty_file(notebook, tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text('{}\ninvalid\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"train.jsonl:2"):
        notebook[0]["load_jsonl"](path)
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="aucun document"):
        notebook[0]["load_jsonl"](path)


@pytest.mark.parametrize("labels", [["O", "B-PER"], ["O", "O"], ["B-PER", "I-PER"]])
def test_invalid_label_inventory(notebook, tmp_path, labels):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(labels), encoding="utf-8")
    with pytest.raises(ValueError):
        notebook[0]["load_labels"](path)


def test_subtokens_adjacent_entities_and_punctuation(notebook):
    text = "محمدعلي."
    entities = [
        {"text": "محمد", "label": "PER", "start": 0, "end": 4},
        {"text": "علي", "label": "PER", "start": 4, "end": 7},
    ]
    labels, owners = notebook[0]["align_spans"](
        text, entities, [(0, 2), (2, 4), (4, 5), (5, 7), (7, 8)],
        {"O": 0, "B-PER": 1, "I-PER": 2},
    )
    assert labels == [1, 2, 1, 2, 0]
    assert owners == [0, 0, 1, 1, None]


@pytest.mark.parametrize("offsets", [[(0, 5)], [(0, 3), (4, 5)], [(4, 5)]])
def test_rejects_crossed_missing_or_normalized_boundary(notebook, offsets):
    with pytest.raises(ValueError):
        notebook[0]["align_spans"](
            "محمد.", [{"text": "محمد", "label": "PER", "start": 0, "end": 4}],
            offsets, {"O": 0, "B-PER": 1, "I-PER": 2},
        )


class CharacterTokenizer:
    """Tokenizer déterministe pour tester les fenêtres sans téléchargement."""

    is_fast = True
    model_max_length = 512
    model_input_names = ["input_ids", "attention_mask", "token_type_ids"]
    padding_side = "right"
    cls_token_id = 101
    sep_token_id = 102
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        assert kwargs["truncation"] is False
        return {"input_ids": list(range(10, 10 + len(text))),
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}

    def num_special_tokens_to_add(self, pair):
        return 2

    def build_inputs_with_special_tokens(self, ids):
        return [101, *ids, 102]

    def create_token_type_ids_from_sequences(self, ids):
        return [0] * (len(ids) + 2)

    def pad(self, data, padding, max_length, return_attention_mask):
        count = max_length - len(data["input_ids"])
        result = {**data, "attention_mask": [1] * len(data["input_ids"])}
        return {key: values + [0] * count if self.padding_side == "right"
                else [0] * count + values for key, values in result.items()}


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_windows_preserve_entities_all_tokens_and_mask_specials(notebook, padding_side):
    tokenizer = CharacterTokenizer()
    tokenizer.padding_side = padding_side
    record = {"id": "long", "text": "abcdefgh", "entities": [
        {"text": "cde", "label": "PER", "start": 2, "end": 5},
    ]}
    windows = notebook[0]["tokenize_and_align_document"](
        record, tokenizer, {"O": 0, "B-PER": 1, "I-PER": 2}, max_length=6,
    )
    assert [(w["token_start"], w["token_end"]) for w in windows] == [(0, 2), (2, 6), (6, 8)]
    supervised = [label for window in windows for label in window["labels"] if label != -100]
    assert supervised == [0, 0, 1, 2, 2, 0, 0, 0]
    for window in windows:
        assert len(window["input_ids"]) == len(window["labels"]) == 6
        for token, mask, label in zip(window["input_ids"], window["attention_mask"], window["labels"]):
            if token in (101, 102) or mask == 0:
                assert label == -100


def test_entity_longer_than_window_is_rejected(notebook):
    record = {"id": "long", "text": "abcde", "entities": [
        {"text": "abcde", "label": "PER", "start": 0, "end": 5},
    ]}
    with pytest.raises(ValueError, match="plus longue"):
        notebook[0]["tokenize_and_align_document"](
            record, CharacterTokenizer(), {"O": 0, "B-PER": 1, "I-PER": 2}, max_length=6,
        )


def test_training_metrics_ignore_specials_and_require_complete_entities(notebook):
    np = pytest.importorskip("numpy")
    metrics = pytest.importorskip("seqeval.metrics")
    scheme = pytest.importorskip("seqeval.scheme")
    namespace, cells = notebook
    namespace.update(np=np, id2label={0: "O", 1: "B-PER", 2: "I-PER"}, IOB2=scheme.IOB2,
                     TRAINING_ENTITY_TYPES=("PER", "COURT", "ORG", "LOC"))
    namespace.update({name: getattr(metrics, name) for name in (
        "precision_score", "recall_score", "f1_score", "accuracy_score",
    )})
    exec(cells["training-metrics"], namespace)
    reference = np.array([[-100, 1, 2, 0, -100]])
    logits = np.eye(3)[np.array([[2, 1, 2, 0, 1]])]
    assert namespace["compute_ner_metrics"]((logits, reference))["f1"] == 1.0
    assert namespace["compute_ner_metrics"]((logits, reference))["macro_f1"] == 0.25
    # Une entité commençant par I-PER ne doit pas recevoir un F1 parfait.
    logits[0, 1] = [0, 0, 1]
    assert namespace["compute_ner_metrics"](((logits,), reference))["f1"] == 0.0
    with pytest.raises(ValueError, match="aucun token"):
        namespace["compute_ner_metrics"]((logits, np.full_like(reference, -100)))


@pytest.mark.parametrize("legacy", [False, True])
def test_trainer_uses_validation_for_best_checkpoint(notebook, tmp_path, legacy):
    namespace, cells = notebook

    class FakeModel:
        @staticmethod
        def from_pretrained(name, **kwargs):
            return SimpleNamespace(config=SimpleNamespace(max_position_embeddings=512))

    class ModernArguments:
        def __init__(self, eval_strategy=None, **kwargs):
            self.evaluation = eval_strategy
            self.options = kwargs

    class LegacyArguments:
        def __init__(self, evaluation_strategy=None, **kwargs):
            self.evaluation = evaluation_strategy
            self.options = kwargs

    class ModernTrainer:
        def __init__(self, processing_class=None, **kwargs):
            self.processor = processing_class
            self.options = kwargs

    class LegacyTrainer:
        def __init__(self, tokenizer=None, **kwargs):
            self.processor = tokenizer
            self.options = kwargs

    data = {"train": [object()], "validation": [object()], "test": [object()]}
    namespace.update(
        inspect=inspect, AutoModelForTokenClassification=FakeModel,
        TrainingArguments=LegacyArguments if legacy else ModernArguments,
        Trainer=LegacyTrainer if legacy else ModernTrainer,
        MODEL_NAME="local-test", labels=["O", "B-PER", "I-PER"],
        id2label={0: "O", 1: "B-PER", 2: "I-PER"},
        label2id={"O": 0, "B-PER": 1, "I-PER": 2},
        MAX_LENGTH=512, RUN_DIR=tmp_path, NUM_EPOCHS=5, LEARNING_RATE=2e-5,
        TRAIN_BATCH_SIZE=4, EVAL_BATCH_SIZE=4, GRADIENT_ACCUMULATION_STEPS=4,
        WEIGHT_DECAY=0.01, SEED=42, model_datasets=data,
        default_data_collator=object(), compute_ner_metrics=object(), tokenizer=object(),
    )
    exec(cells["training-build"], namespace)
    trainer = namespace["trainer"]
    assert trainer.options["train_dataset"] is data["train"]
    assert trainer.options["eval_dataset"] is data["validation"]
    args = trainer.options["args"]
    assert args.evaluation == args.options["save_strategy"] == "epoch"
    assert args.options["load_best_model_at_end"] is True
    assert args.options["metric_for_best_model"] == "macro_f1"
    assert args.options["greater_is_better"] is True
    assert trainer.processor is namespace["tokenizer"]


def test_macro_f1_weights_entity_types_equally(notebook):
    np = pytest.importorskip("numpy")
    metrics = pytest.importorskip("seqeval.metrics")
    scheme = pytest.importorskip("seqeval.scheme")
    namespace, cells = notebook
    namespace.update(np=np, IOB2=scheme.IOB2,
                     TRAINING_ENTITY_TYPES=("PER", "COURT", "ORG", "LOC"),
                     id2label={0: "O", 1: "B-PER", 2: "B-COURT", 3: "B-ORG", 4: "B-LOC"})
    namespace.update({name: getattr(metrics, name) for name in (
        "precision_score", "recall_score", "f1_score", "accuracy_score",
    )})
    exec(cells["training-metrics"], namespace)
    reference = np.array([[-100, 1, 1, 1, 2, 3, 4, -100]])
    logits = np.eye(5)[np.array([[0, 1, 1, 1, 0, 0, 0, 0]])]
    result = namespace["compute_ner_metrics"]((logits, reference))
    assert result["macro_f1"] == 0.25
    assert result["f1"] == pytest.approx(2 / 3)


def test_augmented_training_starts_fresh_and_does_not_use_test(notebook):
    _, cells = notebook
    config = cells["training-config"]
    assert 'MODEL_NAME == "aubmindlab/bert-base-arabertv02"' in config
    assert '"train_augmented.jsonl"' in config
    assert '"validation.jsonl"' in config
    assert 'RUN_DIR.mkdir(parents=True, exist_ok=False)' in config
    assert '"augmented-"' in config
    assert "resume_from_checkpoint=False" in cells["training-run"]
    for cell_id in ("training-config", "training-build", "training-run"):
        assert 'datasets["test"]' not in cells[cell_id]
        assert '"test.jsonl"' not in cells[cell_id]
    assert '"selection_metric": "eval_macro_f1"' in cells["training-export"]
    assert '"data_sha256": training_data_sha256' in cells["training-export"]


@pytest.fixture
def reporting(notebook):
    np = pytest.importorskip("numpy")
    metrics = pytest.importorskip("seqeval.metrics")
    scheme = pytest.importorskip("seqeval.scheme")
    namespace, cells = notebook
    namespace.update(np=np, IOB2=scheme.IOB2)
    namespace.update({name: getattr(metrics, name) for name in (
        "precision_score", "recall_score", "f1_score", "accuracy_score",
    )})
    exec(cells["evaluation-functions"], namespace)
    return namespace, cells


def test_per_entity_report_masks_specials_and_counts_wrong_types(reporting):
    namespace, _ = reporting
    np = namespace["np"]
    labels = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-LOC", 4: "I-LOC",
              5: "B-ORG", 6: "I-ORG"}
    reference = np.array([[-100, 1, 2, 0, 3, -100]])
    logits = np.eye(7)[np.array([[5, 1, 2, 0, 1, 5]])]
    report = namespace["build_ner_report"]((logits,), reference, labels)
    assert report["per_entity"]["PER"]["precision"] == 0.5
    assert report["per_entity"]["PER"]["recall"] == 1.0
    assert report["per_entity"]["LOC"]["support"] == 1
    assert report["per_entity"]["LOC"]["f1"] == 0.0
    assert report["per_entity"]["ORG"] == {
        "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0,
    }
    assert report["micro"]["f1"] == 0.5
    assert report["macro"]["f1"] == pytest.approx(2 / 9)
    assert report["evaluated_tokens"] == 4


def test_empty_entity_report_and_invalid_predictions(reporting):
    namespace, _ = reporting
    np = namespace["np"]
    labels = {0: "O", 1: "B-PER", 2: "I-PER"}
    reference = np.array([[-100, 0, -100]])
    logits = np.eye(3)[np.array([[1, 0, 1]])]
    report = namespace["build_ner_report"](logits, reference, labels)
    assert report["micro"]["f1"] == report["macro"]["f1"] == 0.0
    assert report["per_entity"]["PER"]["support"] == 0
    assert report["token_accuracy"] == 1.0
    with pytest.raises(ValueError, match="aucun token"):
        namespace["build_ner_report"](logits, np.full_like(reference, -100), labels)
    logits[0, 1, 0] = np.nan
    with pytest.raises(ValueError, match="non finis"):
        namespace["build_ner_report"](logits, reference, labels)


def test_evaluation_export_writes_report_and_portable_metadata(notebook, tmp_path):
    namespace, cells = notebook
    model_dir = tmp_path / "best_model"
    model_dir.mkdir()
    (tmp_path / "test.jsonl").write_text('{}\n', encoding="utf-8")
    saved_metrics = []
    namespace.update(
        RUN_DIR=tmp_path, DATA_DIR=tmp_path, BEST_MODEL_DIR=model_dir,
        best_checkpoint="checkpoint-42", MODEL_NAME="local-test", MAX_LENGTH=512,
        label2id={"O": 0, "B-PER": 1, "I-PER": 2},
        id2label={0: "O", 1: "B-PER", 2: "I-PER"},
        test_report={"per_entity": {"PER": {"f1": 0.5}}},
        test_rows=[{"entité": "PER", "precision": 0.5, "recall": 0.5,
                    "f1": 0.5, "support": 2}],
        test_prediction=SimpleNamespace(metrics={"test_f1": 0.5}),
        trainer=SimpleNamespace(save_metrics=lambda *args: saved_metrics.append(args)),
    )
    exec(cells["evaluation-export"], namespace)
    report = json.loads((tmp_path / "test_ner_report.json").read_text(encoding="utf-8"))
    assert report["selection_split"] == "validation"
    assert report["split"] == "test"
    assert report["selected_checkpoint"] == "checkpoint-42"
    assert report["test_sha256"] == hashlib.sha256(b'{}\n').hexdigest()
    assert "PER" in (tmp_path / "test_ner_report.csv").read_text(encoding="utf-8")
    metadata = json.loads((model_dir / "ner_metadata.json").read_text(encoding="utf-8"))
    assert metadata["special_and_padding_label"] == -100
    assert metadata["label2id"]["B-PER"] == 1
    assert saved_metrics == [("test", {"test_f1": 0.5})]


def test_error_analysis_counts_boundaries_types_and_preserves_inputs(reporting):
    namespace, cells = reporting
    exec(cells["error-analysis-functions"], namespace)
    np = namespace["np"]
    mapping = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-LOC", 4: "I-LOC",
               5: "B-ORG", 6: "I-ORG", 7: "B-COURT", 8: "I-COURT"}
    reference = np.array([[-100, 1, 2, 3, 5, -100]])
    logits = np.eye(9)[np.array([[7, 1, 0, 3, 7, 7]])]
    records = [{"id": "doc", "text": "ab cd ef"}]
    windows = [{"document_id": "doc", "window_index": 0,
                "labels": reference[0].tolist(),
                "offset_mapping": [(0, 0), (0, 1), (1, 2), (3, 5), (6, 8), (0, 0)]}]
    original = deepcopy((windows, records))
    rows, examples = namespace["analyze_ner_errors"](
        (logits,), reference, windows, records, mapping, ("PER", "COURT", "ORG", "LOC"),
    )
    counts = {row["type"]: row for row in rows}
    assert counts["PER"]["FP"] == counts["PER"]["FN"] == 1
    assert counts["COURT"]["FP"] == 1
    assert counts["ORG"]["FN"] == 1
    assert counts["LOC"]["TP"] == 1
    assert len(examples) == 4
    per_fn = next(row for row in examples if row["type"] == "PER" and row["erreur"] == "FN")
    assert per_fn["texte"] == "ab"
    assert per_fn["BIO réel"] == "B-PER I-PER"
    assert per_fn["BIO prédit"] == "B-PER O"
    court_fp = next(row for row in examples if row["type"] == "COURT")
    assert "ORG [6, 8)" in court_fp["entités réelles"]
    assert (windows, records) == original
    report = namespace["build_ner_report"](logits, reference, mapping)
    for row in rows:
        metrics = report["per_entity"][row["type"]]
        assert row["TP"] + row["FN"] == metrics["support"]
        denominator = 2 * row["TP"] + row["FP"] + row["FN"]
        assert (2 * row["TP"] / denominator if denominator else 0) == metrics["f1"]


def test_error_analysis_rejects_stale_windows_and_does_not_repair_bio(reporting):
    namespace, cells = reporting
    exec(cells["error-analysis-functions"], namespace)
    np = namespace["np"]
    mapping = {0: "O", 1: "B-PER", 2: "I-PER"}
    reference = np.array([[-100, 1, -100]])
    logits = np.eye(3)[np.array([[1, 2, 1]])]
    windows = [{"document_id": "doc", "window_index": 0,
                "labels": [-100, 1, -100], "offset_mapping": [(0, 0), (0, 1), (0, 0)]}]
    records = [{"id": "doc", "text": "a"}]
    rows, examples = namespace["analyze_ner_errors"](
        logits, reference, windows, records, mapping, ("PER",),
    )
    assert rows[0]["TP"] == rows[0]["FP"] == 0
    assert rows[0]["FN"] == 1
    assert examples[0]["BIO prédit"] == "I-PER"
    windows[0]["labels"][1] = 0
    with pytest.raises(ValueError, match="désynchronisés"):
        namespace["analyze_ner_errors"](logits, reference, windows, records, mapping, ("PER",))

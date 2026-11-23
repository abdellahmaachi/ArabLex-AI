"""Inférence et chargement NER avec un minuscule checkpoint local, sans Hub."""

import json
from types import SimpleNamespace

import pytest

from src.ner.annotator import AraBERTAnnotator


@pytest.fixture
def checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    vocab = {token: index for index, token in enumerate(
        ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "محمد", "علي",
         "محكمة", "النقض", ".", "حَكَمَ"]
    )}
    backend = tokenizers.Tokenizer(tokenizers.models.WordPiece(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
        cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]", model_max_length=12,
    )
    tokenizer.save_pretrained(tmp_path)
    labels = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-COURT", 4: "I-COURT"}
    config = transformers.BertConfig(
        vocab_size=len(vocab), hidden_size=8, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=16, max_position_embeddings=12,
        id2label=labels, label2id={label: index for index, label in labels.items()},
    )
    model = transformers.BertForTokenClassification(config)
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.zero_()
        model.classifier.bias[0] = 1
    model.save_pretrained(tmp_path)
    return tmp_path


def deterministic_model(annotator):
    torch = annotator._torch

    class Model(torch.nn.Module):
        def forward(self, input_ids, attention_mask, **kwargs):
            assert not self.training
            assert not torch.is_grad_enabled()
            assert input_ids.shape == attention_mask.shape
            logits = torch.zeros((*input_ids.shape, 5), device=input_ids.device)
            # Inclut des labels pour CLS/SEP : ils doivent être ignorés.
            mapping = {2: 1, 3: 1, 5: 1, 6: 2, 7: 3, 8: 4}
            for position, token_id in enumerate(input_ids[0].tolist()):
                logits[0, position, mapping.get(token_id, 0)] = 10
            return SimpleNamespace(logits=logits)

    annotator.model = Model()


def test_local_checkpoint_loads_and_predicts(checkpoint):
    annotator = AraBERTAnnotator(checkpoint)
    assert not annotator.model.training
    assert str(annotator.device) == "cpu"
    assert annotator.annotate("محمد") == []  # Poids locaux : classe O dominante.


def test_exact_arabic_offsets_repetitions_and_special_tokens(checkpoint):
    annotator = AraBERTAnnotator(checkpoint)
    deterministic_model(annotator)
    text = "حَكَمَ محمد  علي. محكمة النقض محمد"
    first, court, repeated = text.index("محمد"), text.index("محكمة"), text.rindex("محمد")
    entities = annotator.annotate(text)
    assert entities == [
        {"text": "محمد  علي", "label": "PER", "start": first, "end": text.index(".")},
        {"text": "محكمة النقض", "label": "COURT", "start": court, "end": repeated - 1},
        {"text": "محمد", "label": "PER", "start": repeated, "end": len(text)},
    ]
    assert all(entity["text"] == text[entity["start"]:entity["end"]] for entity in entities)


def test_long_text_is_not_truncated_and_cross_window_entity_is_decoded_once(checkpoint):
    annotator = AraBERTAnnotator(checkpoint, max_length=5, stride=1)
    deterministic_model(annotator)
    text = ". . محمد علي . . . محمد علي"
    entities = annotator.annotate(text)
    assert [entity["text"] for entity in entities] == ["محمد علي", "محمد علي"]
    assert entities[0]["start"] == text.index("محمد")
    assert entities[-1]["end"] == len(text)


def test_orphan_inside_tag_can_be_ignored_or_rejected(checkpoint):
    annotator = AraBERTAnnotator(checkpoint)
    deterministic_model(annotator)
    assert annotator.annotate("علي") == []
    annotator.invalid_bio = "raise"
    with pytest.raises(ValueError, match="Transition"):
        annotator.annotate("علي")


def test_empty_text_and_invalid_input(checkpoint):
    annotator = AraBERTAnnotator(checkpoint)
    assert annotator.annotate("") == annotator.annotate(" \n\t") == []
    with pytest.raises(TypeError, match="text"):
        annotator.annotate(None)


@pytest.mark.parametrize("options", [
    {"max_length": 2}, {"max_length": 13}, {"max_length": True},
    {"stride": -1}, {"max_length": 5, "stride": 3}, {"invalid_bio": "repair"},
])
def test_invalid_options(checkpoint, options):
    with pytest.raises(ValueError):
        AraBERTAnnotator(checkpoint, **options)


def test_invalid_preprocessing_and_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        AraBERTAnnotator(tmp_path / "missing")
    (tmp_path / "ner_metadata.json").write_text(
        json.dumps({"preprocessing": "normalized"}), encoding="utf-8",
    )
    with pytest.raises(ValueError, match="prétraitement"):
        AraBERTAnnotator(tmp_path)


def test_rejects_non_bio_model_labels(checkpoint):
    path = checkpoint / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["id2label"]["1"] = "LABEL_1"
    config["label2id"] = {label: int(index) for index, label in config["id2label"].items()}
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="BIO"):
        AraBERTAnnotator(checkpoint)

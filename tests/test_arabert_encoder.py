"""Tests hors ligne avec un minuscule checkpoint BERT aléatoire local."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.rag.arabert_encoder import AraBERTEncoder


@pytest.fixture
def checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "قانون", "محكمة", "لا"]
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(vocabulary) + "\n", encoding="utf-8")
    tokenizer = transformers.BertTokenizer(vocab_file=str(vocab), do_lower_case=False)
    tokenizer.save_pretrained(tmp_path)
    with torch.random.fork_rng():
        torch.manual_seed(7)
        model = transformers.BertModel(transformers.BertConfig(
            vocab_size=len(vocabulary), hidden_size=12, num_hidden_layers=1,
            num_attention_heads=3, intermediate_size=16, max_position_embeddings=16,
        ))
    model.save_pretrained(tmp_path)
    return tmp_path


def test_shapes_norms_order_and_batch_independence(checkpoint):
    encoder = AraBERTEncoder(checkpoint, batch_size=2)
    texts = ["قانون", "محكمة لا قانون", "لا محكمة"]
    vectors = encoder.encode_documents(iter(texts))
    assert vectors.shape == (3, 12)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
    singles = np.stack([encoder.encode_query(text) for text in texts])
    np.testing.assert_allclose(vectors, singles, atol=1e-6)
    np.testing.assert_array_equal(vectors, encoder.encode(texts))
    assert encoder.encode([]).shape == (0, 12)
    assert not encoder.model.training


def test_mean_pooling_excludes_special_tokens_and_padding(checkpoint):
    encoder = AraBERTEncoder(checkpoint)
    torch = encoder._torch
    texts = ["قانون", "محكمة لا قانون"]
    inputs = encoder.tokenizer(texts, padding=True, return_tensors="pt")
    with torch.inference_mode():
        hidden = encoder.model(**inputs).last_hidden_state
        expected = torch.stack([hidden[0, 1], hidden[1, 1:4].mean(dim=0)])
        expected = expected / torch.linalg.vector_norm(expected, dim=1, keepdim=True)
    np.testing.assert_allclose(encoder.encode(texts), expected.numpy(), atol=1e-6)


def test_length_limit_and_truncation(checkpoint):
    encoder = AraBERTEncoder(checkpoint, max_length=4)
    np.testing.assert_array_equal(
        encoder.encode_query("قانون محكمة لا"), encoder.encode_query("قانون محكمة"),
    )
    strict = AraBERTEncoder(checkpoint, max_length=4, truncate=False)
    assert strict.encode_query("قانون محكمة").shape == (12,)
    with pytest.raises(ValueError, match="trop long"):
        strict.encode_query("قانون محكمة لا")
    with pytest.raises(ValueError, match="max_length"):
        AraBERTEncoder(checkpoint, max_length=17)


def test_invalid_texts(checkpoint):
    encoder = AraBERTEncoder(checkpoint)
    for texts, error in [("قانون", TypeError), ([None], TypeError), (["  "], ValueError)]:
        with pytest.raises(error):
            encoder.encode(texts)


def test_invalid_initialization(tmp_path):
    with pytest.raises(FileNotFoundError):
        AraBERTEncoder(tmp_path / "missing")
    for size in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="batch_size"):
            AraBERTEncoder(tmp_path, batch_size=size)
    with pytest.raises(TypeError, match="truncate"):
        AraBERTEncoder(tmp_path, truncate="yes")


def test_checkpoint_with_tuple_output_configuration(checkpoint):
    encoder = AraBERTEncoder(checkpoint)
    expected = encoder.encode_query("قانون")
    encoder.model.config.return_dict = False
    np.testing.assert_array_equal(encoder.encode_query("قانون"), expected)


@pytest.mark.parametrize("value,message", [(0.0, "nul"), (float("nan"), "non finis"),
                                         (float("inf"), "non finis")])
def test_rejects_invalid_model_representations(checkpoint, monkeypatch, value, message):
    encoder = AraBERTEncoder(checkpoint)

    def forward(input_ids, **kwargs):
        hidden = encoder._torch.full(
            (*input_ids.shape, encoder.dimension), value, device=input_ids.device,
        )
        return SimpleNamespace(last_hidden_state=hidden)

    monkeypatch.setattr(encoder.model, "forward", forward)
    with pytest.raises(ValueError, match=message):
        encoder.encode_query("قانون")

"""Validation BIO et fidélité des positions dans le texte source."""

import pytest

from src.ner.bio_decoder import decode_bio, validate_bio_labels


@pytest.mark.parametrize("labels", [
    [], ["O"], ["B-PER"], ["B-PER", "I-PER", "O"],
    ["B-PER", "B-PER"], ["B-LEGAL-REF", "I-LEGAL-REF"],
])
def test_valid_bio(labels):
    assert validate_bio_labels(labels) is None


@pytest.mark.parametrize("labels", [
    ["I-PER"], ["O", "I-PER"], ["B-PER", "I-ORG"],
    ["B-PER", "O", "I-PER"], [""], ["B-"], ["I-"], ["PER"],
    ["S-PER"], ["b-PER"], ["O-PER"], ["B- PER"], ["B-PER\n"], [None], [1],
])
def test_invalid_bio(labels):
    with pytest.raises(ValueError, match="indice"):
        validate_bio_labels(labels)


def test_arabic_offsets_preserve_diacritics_whitespace_and_subwords():
    text = "⚖ حَكَمَ محمد  بن\tعلي، محمد"
    first = text.index("محمد")
    middle = text.index("بن")
    last = text.index("علي")
    repeated = text.rindex("محمد")
    labels = ["O", "B-PER", "I-PER", "I-PER", "I-PER", "O", "B-PER"]
    offsets = [(2, first - 1), (first, first + 2), (first + 2, first + 4),
               (middle, middle + 2), (last, last + 3), (last + 3, last + 4),
               (repeated, len(text))]
    assert decode_bio(text, labels, offsets) == [
        {"text": "محمد  بن\tعلي", "label": "PER", "start": first, "end": last + 3},
        {"text": "محمد", "label": "PER", "start": repeated, "end": len(text)},
    ]
    assert decode_bio(text, ["B-ACT"], [(2, first - 1)])[0]["text"] == "حَكَمَ"


def test_consecutive_beginnings_and_final_entity():
    assert decode_bio("a b c", ["B-X", "B-X", "B-Y"], [(0, 1), (2, 3), (4, 5)]) == [
        {"text": "a", "label": "X", "start": 0, "end": 1},
        {"text": "b", "label": "X", "start": 2, "end": 3},
        {"text": "c", "label": "Y", "start": 4, "end": 5},
    ]


def test_empty_and_outside_sequences():
    assert decode_bio("", [], []) == []
    assert decode_bio("abc", ["O"], [(0, 3)]) == []


@pytest.mark.parametrize("offsets", [
    [(-1, 1)], [(0, 4)], [(0, 0)], [(2, 1)], [(0.0, 1)], [(False, 1)],
    [(0,)], [(0, 1, 2)], [None], ["01"],
    [(0, 2), (1, 3)], [(2, 3), (0, 1)],
])
def test_invalid_offsets_even_for_outside_tokens(offsets):
    with pytest.raises(ValueError, match="Offset"):
        decode_bio("abc", ["O"] * len(offsets), offsets)


def test_mismatched_lengths():
    with pytest.raises(ValueError, match="longueur"):
        decode_bio("abc", ["B-X"], [])
    with pytest.raises(ValueError, match="longueur"):
        decode_bio("abc", [], [(0, 3)])


def test_decoder_validates_transitions():
    with pytest.raises(ValueError, match="Transition"):
        decode_bio("abc", ["I-X"], [(0, 3)])


def test_rejects_bare_label_string_and_non_text():
    with pytest.raises(TypeError, match="labels"):
        validate_bio_labels("O")
    with pytest.raises(TypeError, match="text"):
        decode_bio(None, [], [])

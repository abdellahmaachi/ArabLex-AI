"""Validation BIO (IOB2) et extraction depuis le texte original."""

from collections.abc import Sequence
from numbers import Integral
from typing import TypedDict


class Entity(TypedDict):
    """Entité avec indices de caractères Python, fin exclusive."""

    text: str
    label: str
    start: int
    end: int


def validate_bio_labels(labels: Sequence[str]) -> None:
    """Lève ValueError pour une étiquette ou transition BIO invalide.

    Accepte O, B-TYPE et I-TYPE, avec TYPE non vide sans espace.
    I-TYPE doit suivre B-TYPE ou I-TYPE du même type. Deux B successifs
    ouvrent deux entités distinctes. Une séquence vide est valide.
    """
    if isinstance(labels, (str, bytes)):
        raise TypeError("labels doit être une séquence d'étiquettes")
    active_type = None
    for index, label in enumerate(labels):
        if not isinstance(label, str):
            raise ValueError(f"Étiquette BIO invalide à l'indice {index}: {label!r}")
        if label == "O":
            active_type = None
            continue
        prefix, separator, entity_type = label.partition("-")
        if (
            prefix not in {"B", "I"}
            or not separator
            or not entity_type
            or any(character.isspace() for character in entity_type)
        ):
            raise ValueError(f"Étiquette BIO invalide à l'indice {index}: {label!r}")
        if prefix == "I" and active_type != entity_type:
            raise ValueError(f"Transition BIO invalide à l'indice {index}: {label!r}")
        active_type = entity_type


def decode_bio(
    text: str,
    labels: Sequence[str],
    offsets: Sequence[tuple[int, int]],
) -> list[Entity]:
    """Convertit les labels de tokens en entités {text, label, start, end}.

    Les offsets [start, end) pointent dans le texte original non normalisé,
    en caractères Python (ni octets ni unités UTF-16). Ils doivent être
    non vides, ordonnés, sans chevauchement et aussi nombreux que les labels.
    Retirer les tokens spéciaux/padding et leurs labels avant cet appel :
    leurs offsets vides, souvent (0, 0), sont rejetés.

    Le slicing préserve espaces, ponctuation et diacritiques entre tokens,
    y compris les sous-mots. Aucune correction silencieuse n'est appliquée.
    Une entrée invalide lève ValueError (TypeError pour un texte non chaîne).
    """
    if not isinstance(text, str):
        raise TypeError("text doit être une chaîne")
    validate_bio_labels(labels)
    if len(labels) != len(offsets):
        raise ValueError("labels et offsets doivent avoir la même longueur")
    spans: list[tuple[int, int]] = []
    previous_end = 0
    for index, offset in enumerate(offsets):
        if isinstance(offset, (str, bytes)):
            raise ValueError(f"Offset invalide à l'indice {index}: {offset!r}")
        try:
            start, end = offset
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Offset invalide à l'indice {index}: {offset!r}") from exc
        if any(isinstance(v, bool) or not isinstance(v, Integral) for v in (start, end)):
            raise ValueError(f"Offset non entier à l'indice {index}: {offset!r}")
        if not (previous_end <= start < end <= len(text)):
            raise ValueError(f"Offset hors limites ou chevauchant à l'indice {index}: {offset!r}")
        spans.append((int(start), int(end)))
        previous_end = end

    entities: list[Entity] = []
    active: Entity | None = None
    for label, (start, end) in zip(labels, spans):
        if label == "O" or label.startswith("B-"):
            if active is not None:
                active["text"] = text[active["start"] : active["end"]]
                entities.append(active)
                active = None
        if label.startswith("B-"):
            active = Entity(text="", label=label[2:], start=start, end=end)
        elif label.startswith("I-"):
            # La validation garantit une entité active du même type.
            assert active is not None
            active["end"] = end
    if active is not None:
        active["text"] = text[active["start"] : active["end"]]
        entities.append(active)
    return entities

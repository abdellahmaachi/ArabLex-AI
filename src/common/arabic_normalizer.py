"""Normalisation configurable du texte arabe, sans mutation des documents.

Les transformations peuvent perdre de l'information : conserver ``document["text"]``
et utiliser séparément la chaîne renvoyée par :func:`normalize_arabic`.
"""

import re
import unicodedata
from dataclasses import dataclass

_DIACRITICS = re.compile("[\u064b-\u0652\u0670]")
_WHITESPACE = re.compile(r"\s+")
_ALEF_TABLE = str.maketrans("أإآٱ", "اااا")
_HAMZA_TABLE = str.maketrans("ؤئ", "وي")
_DIGIT_TABLE = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


@dataclass(frozen=True)
class NormalizationConfig:
    """Options indépendantes de normalisation.

    ``normalize_unicode`` applique NFC pour composer notamment les alifs avec
    hamza décomposée. Les conversions de lettres supplémentaires, des chiffres
    et des espaces sont désactivées par défaut.
    """

    normalize_unicode: bool = True
    remove_diacritics: bool = True
    remove_tatweel: bool = True
    normalize_alef: bool = True
    normalize_alef_maqsura: bool = False
    normalize_teh_marbuta: bool = False
    normalize_hamza: bool = False
    normalize_digits: bool = False
    normalize_whitespace: bool = False


def strip_diacritics(text: str) -> str:
    """Retire U+064B–U+0652 et l'alif suscrit U+0670.

    Les hamzas combinatoires et autres marques coraniques sont conservées.
    """
    return _DIACRITICS.sub("", text)


def strip_tatweel(text: str) -> str:
    """Retire le caractère d'allongement tatweel (ـ, U+0640)."""
    return text.replace("\u0640", "")


def normalize_arabic(text: str, config: NormalizationConfig | None = None) -> str:
    """Renvoie une chaîne normalisée selon ``config``, sans modifier l'original.

    Conversions optionnelles : ``ى → ي``, ``ة → ه``, ``ؤ → و`` et ``ئ → ي``.
    La hamza isolée ``ء`` est conservée. Les deux séries de chiffres arabes
    peuvent être converties en chiffres ASCII. La normalisation des espaces
    remplace les suites d'espaces (y compris les sauts de ligne) par un espace
    et supprime les espaces aux extrémités. La ponctuation est conservée.

    Exemple : ``normalized_text = normalize_arabic(document["text"])``.
    """
    if not isinstance(text, str):
        raise TypeError("text doit être une chaîne de caractères")
    if config is None:
        config = NormalizationConfig()

    result = unicodedata.normalize("NFC", text) if config.normalize_unicode else text
    if config.remove_diacritics:
        result = strip_diacritics(result)
    if config.remove_tatweel:
        result = strip_tatweel(result)
    if config.normalize_alef:
        result = result.translate(_ALEF_TABLE)
    if config.normalize_alef_maqsura:
        result = result.replace("ى", "ي")
    if config.normalize_teh_marbuta:
        result = result.replace("ة", "ه")
    if config.normalize_hamza:
        result = result.translate(_HAMZA_TABLE)
    if config.normalize_digits:
        result = result.translate(_DIGIT_TABLE)
    if config.normalize_whitespace:
        result = _WHITESPACE.sub(" ", result).strip()
    return result

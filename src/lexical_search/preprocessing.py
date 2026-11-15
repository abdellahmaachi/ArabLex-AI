"""Prétraitement lexical arabe sans modification du texte source.

Ordre : normalisation, nettoyage/tokenisation, stopwords, racinisation légère.
La liste locale de stopwords est volontairement limitée et conserve les négations
(لا، لم، لن، ليس). Aucun téléchargement de corpus NLTK n'est nécessaire.
"""

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field

from nltk.stem.snowball import ArabicStemmer

from src.common.arabic_normalizer import NormalizationConfig, normalize_arabic

DEFAULT_STOPWORDS = frozenset(
    "في من إلى على عن مع و ف ثم أو أم بل إن أن كأن لكن لعل ليت "
    "هذا هذه هذان هاتان هؤلاء ذلك تلك أولئك الذي التي الذين اللاتي اللواتي "
    "هو هي هما هم هن أنا نحن أنت أنتم أنتن كان كانت يكون تكون "
    "قد لقد إذ إذا حيث هنا هناك كما أيضا بين عند لدى خلال".split()
)
_TOKEN_PATTERN = re.compile(r"\d+|[^\d\s]+")


@dataclass(frozen=True)
class PreprocessingConfig:
    """Options communes à appliquer aux documents et aux requêtes.

    Les nombres sont conservés par défaut pour les références juridiques.
    ``min_token_length`` s'applique aux tokens finaux, nombres compris.
    Désactiver ``stemming`` permet de conserver les mots normalisés complets.
    """

    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    remove_stopwords: bool = True
    stemming: bool = True
    keep_numbers: bool = True
    min_token_length: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.min_token_length, bool) or not isinstance(self.min_token_length, int):
            raise TypeError("min_token_length doit être un entier")
        if self.min_token_length < 1:
            raise ValueError("min_token_length doit être supérieur ou égal à 1")


def clean_text(text: str, *, keep_numbers: bool = True) -> str:
    """Conserve lettres/marques arabes et, en option, chiffres Unicode.

    La ponctuation, les symboles et les caractères non arabes deviennent des
    séparateurs, afin de ne pas fusionner deux mots. Les espaces sont regroupés.
    Les caractères de formatage invisibles sont retirés. Les formes de
    présentation arabes sont décomposées (NFKC) pour traiter les ligatures.
    Ce nettoyage est destiné au texte brut, pas à l'analyse de pages HTML.
    """
    if not isinstance(text, str):
        raise TypeError("text doit être une chaîne de caractères")
    characters: list[str] = []
    for character in unicodedata.normalize("NFKC", text):
        category = unicodedata.category(character)
        if category == "Cf":
            continue
        is_arabic = (
            category[0] in {"L", "M"}
            and "ARABIC" in unicodedata.name(character, "")
        )
        if is_arabic or (keep_numbers and category == "Nd"):
            characters.append(character)
        else:
            characters.append(" ")
    return " ".join("".join(characters).split())


def tokenize(text: str, *, keep_numbers: bool = True) -> list[str]:
    """Découpe le texte nettoyé ; les nombres et les mots sont séparés.

    Les diacritiques restent attachés aux mots. Les marques isolées sont ignorées.
    L'ordre et les répétitions sont conservés pour les fréquences lexicales.
    """
    cleaned = clean_text(text, keep_numbers=keep_numbers)
    return [
        token
        for token in _TOKEN_PATTERN.findall(cleaned)
        if any(unicodedata.category(character)[0] in {"L", "N"} for character in token)
    ]


def _normalized_tokens(
    text: str, normalization: NormalizationConfig, *, keep_numbers: bool = True
) -> list[str]:
    # Décomposer les ligatures avant la normalisation des lettres et diacritiques.
    return tokenize(
        normalize_arabic(unicodedata.normalize("NFKC", text), normalization),
        keep_numbers=keep_numbers,
    )


def remove_stopwords(
    tokens: Iterable[str],
    stopwords: Iterable[str] = DEFAULT_STOPWORDS,
    *,
    normalization: NormalizationConfig | None = None,
) -> list[str]:
    """Filtre des tokens déjà normalisés, sans modifier la séquence fournie.

    Les stopwords sont normalisés avec les mêmes options que le texte. Une liste
    personnalisée remplace la liste par défaut ; une liste vide conserve tout.
    """
    config = normalization if normalization is not None else NormalizationConfig()
    excluded = {
        token for word in stopwords for token in _normalized_tokens(word, config)
    }
    return [token for token in tokens if token not in excluded]


def stem_tokens(tokens: Iterable[str]) -> list[str]:
    """Applique Snowball arabe ; renvoie des radicaux, pas des racines garanties.

    Les nombres sont conservés. Le stemmer possède ses propres règles de
    normalisation ; désactiver la racinisation pour contrôler celles-ci uniquement
    via ``NormalizationConfig``.
    """
    stemmer = ArabicStemmer()
    return [token if token.isdecimal() else stemmer.stem(token) for token in tokens]


def preprocess_text(
    text: str,
    config: PreprocessingConfig | None = None,
    *,
    stopwords: Iterable[str] = DEFAULT_STOPWORDS,
) -> list[str]:
    """Renvoie les tokens prêts pour l'indexation ou une requête lexicale.

    Exemple : ``tokens = preprocess_text(document["text"])``. Le document et son
    texte d'origine restent inchangés. Joindre les tokens avec un espace si le
    consommateur attend une chaîne plutôt qu'une liste de tokens.
    """
    if not isinstance(text, str):
        raise TypeError("text doit être une chaîne de caractères")
    options = config if config is not None else PreprocessingConfig()
    tokens = _normalized_tokens(text, options.normalization, keep_numbers=options.keep_numbers)
    if options.remove_stopwords:
        tokens = remove_stopwords(tokens, stopwords, normalization=options.normalization)
    if options.stemming:
        tokens = stem_tokens(tokens)
    return [token for token in tokens if len(token) >= options.min_token_length]

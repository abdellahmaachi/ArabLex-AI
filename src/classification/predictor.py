"""Chargement d'un Naive Bayes sauvegardé et classification de textes arabes."""

from collections import Counter
from collections.abc import Iterable, Mapping
from numbers import Integral
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix

from src.classification.naive_bayes import MultinomialNaiveBayes
from src.lexical_search.preprocessing import (
    DEFAULT_STOPWORDS,
    PreprocessingConfig,
    preprocess_text,
)


class NaiveBayesPredictor:
    """Associe le modèle aux caractéristiques exactes utilisées à l'apprentissage.

    ``vocabulary`` associe chaque terme prétraité à sa colonne de comptage.
    Le modèle doit avoir été entraîné sur ces comptages, avec les mêmes options
    de prétraitement et stopwords. Aucun vocabulaire n'est appris à la prédiction.

    Exemple après l'entraînement ::

        predictor = NaiveBayesPredictor(model, vocabulary, config=config)
        predictor.save("artifacts/naive_bayes/predictor.joblib")
        loaded = NaiveBayesPredictor.load("artifacts/naive_bayes/predictor.joblib")
        label = loaded.predict("عقد ملكية")

    Les mots inconnus sont ignorés. Un texte vide, filtré ou entièrement inconnu
    donne les probabilités a priori du modèle. Une égalité est départagée dans
    l'ordre de ``model.classes_``. Les textes originaux ne sont pas modifiés.
    """

    def __init__(
        self,
        model: MultinomialNaiveBayes,
        vocabulary: Mapping[str, int],
        *,
        config: PreprocessingConfig | None = None,
        stopwords: Iterable[str] = DEFAULT_STOPWORDS,
    ) -> None:
        if not isinstance(model, MultinomialNaiveBayes):
            raise TypeError("model doit être un MultinomialNaiveBayes")
        if not getattr(model, "_fitted", False):
            raise ValueError("Le modèle doit être entraîné avant son utilisation")
        if not isinstance(vocabulary, Mapping) or not vocabulary:
            raise ValueError("Le vocabulaire doit être une correspondance non vide")
        if any(not isinstance(term, str) or not term.strip() for term in vocabulary):
            raise ValueError("Les termes du vocabulaire doivent être des chaînes non vides")
        if any(
            isinstance(column, bool) or not isinstance(column, Integral)
            for column in vocabulary.values()
        ):
            raise ValueError("Les colonnes du vocabulaire doivent être des entiers")
        columns = list(vocabulary.values())
        if sorted(columns) != list(range(len(vocabulary))):
            raise ValueError("Les colonnes doivent être uniques et contiguës à partir de zéro")
        if len(vocabulary) != model.n_features_in_:
            raise ValueError("Le vocabulaire et le modèle doivent avoir le même nombre de colonnes")
        if config is not None and not isinstance(config, PreprocessingConfig):
            raise TypeError("config doit être une PreprocessingConfig")
        if isinstance(stopwords, str):
            raise TypeError("stopwords doit être une collection de mots")
        words = tuple(stopwords)
        if any(not isinstance(word, str) for word in words):
            raise TypeError("Les stopwords doivent être des chaînes")
        self._model = model
        self._vocabulary = {term: int(column) for term, column in vocabulary.items()}
        self._config = config if config is not None else PreprocessingConfig()
        self._stopwords = words

    @property
    def classes_(self) -> list[str | int]:
        """Classes dans l'ordre des probabilités du modèle."""
        return self._model.classes_.tolist()

    @property
    def vocabulary_(self) -> dict[str, int]:
        return dict(self._vocabulary)

    def vectorize(self, texts: Iterable[str]) -> csr_matrix:
        """Transforme des textes en comptages CSR sans ajuster les statistiques."""
        if isinstance(texts, str):
            raise TypeError("Fournir une collection de textes, par exemple [texte]")
        indices: list[int] = []
        counts: list[int] = []
        indptr = [0]
        for text in texts:
            frequencies = Counter(preprocess_text(text, self._config, stopwords=self._stopwords))
            for term, frequency in frequencies.items():
                if term in self._vocabulary:
                    indices.append(self._vocabulary[term])
                    counts.append(frequency)
            indptr.append(len(indices))
        matrix = csr_matrix(
            (counts, indices, indptr),
            shape=(len(indptr) - 1, len(self._vocabulary)), dtype=np.float64,
        )
        matrix.sort_indices()
        return matrix

    def predict(self, text: str) -> str | int:
        """Renvoie la classe prédite pour un texte, avec un type Python natif."""
        return self._model.predict(self.vectorize([text]))[0].item()

    def predict_many(self, texts: Iterable[str]) -> list[str | int]:
        """Prédit un lot de textes dans l'ordre fourni ; un lot vide donne []."""
        return self._model.predict(self.vectorize(texts)).tolist()

    def predict_proba(self, text: str) -> dict[str | int, float]:
        """Renvoie les probabilités a posteriori de chaque classe pour un texte."""
        probabilities = self._model.predict_proba(self.vectorize([text]))[0]
        return dict(zip(self.classes_, probabilities.tolist()))

    def save(self, path: str | Path) -> None:
        """Sauvegarde modèle, vocabulaire et prétraitement dans un seul fichier."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "version": 1,
            "model": self._model,
            "vocabulary": self._vocabulary,
            "config": self._config,
            "stopwords": self._stopwords,
        }, target)

    @classmethod
    def load(cls, path: str | Path) -> "NaiveBayesPredictor":
        """Charge et valide un fichier produit par ``save``, sans réentraînement.

        Charger uniquement des fichiers de confiance : joblib utilise pickle,
        dont la désérialisation peut exécuter du code. Les erreurs de lecture et
        de désérialisation sont propagées. Utiliser un environnement compatible
        avec celui de l'apprentissage.
        """
        payload = joblib.load(path)
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError("Format de sauvegarde du prédicteur non reconnu")
        required = {"model", "vocabulary", "config", "stopwords"}
        if not required.issubset(payload):
            raise ValueError("Sauvegarde incomplète : modèle, vocabulaire ou prétraitement absent")
        if not isinstance(payload["config"], PreprocessingConfig):
            raise ValueError("La sauvegarde doit contenir une configuration de prétraitement")
        return cls(
            payload["model"], payload["vocabulary"],
            config=payload["config"], stopwords=payload["stopwords"],
        )

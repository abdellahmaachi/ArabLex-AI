"""TF-IDF du corpus et des requêtes dans un même espace lexical.

TF = nombre d'occurrences ; IDF = log((1 + N) / (1 + df)) + 1.
La normalisation L2 est activée par défaut. Les matrices retournées sont creuses
(CSR), avec une colonne par terme et une ligne par document ou requête.
"""

from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from src.common.document_schema import Document
from src.lexical_search.indexer import build_inverted_index
from src.lexical_search.preprocessing import (
    DEFAULT_STOPWORDS,
    PreprocessingConfig,
    preprocess_text,
)


class TfidfModel:
    """Calcule et sauvegarde les statistiques nécessaires à la recherche.

    ``fit_transform(documents)`` renvoie la matrice du corpus ; ``document_ids_``
    donne l'identifiant de chaque ligne, dans l'ordre d'entrée. Les documents
    vides contribuent à N et gardent une ligne nulle. Un corpus sans aucun terme
    est refusé. Les textes et la matrice du corpus ne sont pas sauvegardés.

    Exemple ::

        model = TfidfModel()
        matrix = model.fit_transform(documents)
        model.save("artifacts/lexical/tfidf.joblib")
        query = TfidfModel.load("artifacts/lexical/tfidf.joblib").vectorize_query("عقد")
    """

    def __init__(
        self,
        config: PreprocessingConfig | None = None,
        *,
        stopwords: Iterable[str] = DEFAULT_STOPWORDS,
        norm: str | None = "l2",
    ) -> None:
        if norm not in (None, "l2"):
            raise ValueError("norm doit être 'l2' ou None")
        self._config = config if config is not None else PreprocessingConfig()
        self._stopwords = tuple(stopwords)
        self._norm = norm
        self._counts: DictVectorizer | None = None
        self._tfidf: TfidfTransformer | None = None
        self.document_ids_: tuple[str, ...] = ()
        self.n_documents_: int = 0

    def _check_fitted(self) -> None:
        if self._counts is None or self._tfidf is None:
            raise NotFittedError("Appeler fit ou fit_transform avant d'utiliser le modèle TF-IDF")

    @property
    def vocabulary_(self) -> dict[str, int]:
        """Copie de la correspondance terme → colonne de la matrice."""
        self._check_fitted()
        return dict(self._counts.vocabulary_)

    @property
    def idf_(self) -> np.ndarray:
        """Copie des poids IDF dans l'ordre des colonnes."""
        self._check_fitted()
        return self._tfidf.idf_.copy()

    @property
    def document_frequency_(self) -> np.ndarray:
        """Nombre de documents contenant chaque terme, dans l'ordre des colonnes."""
        self._check_fitted()
        return self._document_frequency.copy()

    def fit(self, documents: Iterable[Document]) -> "TfidfModel":
        """Apprend les statistiques du corpus et renvoie le modèle."""
        self.fit_transform(documents)
        return self

    def fit_transform(self, documents: Iterable[Document]) -> csr_matrix:
        """Apprend le vocabulaire/IDF et renvoie la matrice TF-IDF du corpus.

        Un nouvel apprentissage remplace les statistiques précédentes. En cas
        d'erreur, le modèle précédemment appris reste utilisable.
        """
        corpus = list(documents)
        if not corpus:
            raise ValueError("Le corpus ne doit pas être vide")
        index = build_inverted_index(corpus, self._config, stopwords=self._stopwords)
        if not index:
            raise ValueError("Le corpus ne contient aucun terme après prétraitement")

        document_ids = tuple(document["id"] for document in corpus)
        counts_by_id: dict[str, dict[str, int]] = {doc_id: {} for doc_id in document_ids}
        for term, postings in index.items():
            for doc_id, frequency in postings.items():
                counts_by_id[doc_id][term] = frequency

        counts = DictVectorizer(sparse=True, sort=True)
        count_matrix = counts.fit_transform(counts_by_id[doc_id] for doc_id in document_ids)
        tfidf = TfidfTransformer(norm=self._norm, smooth_idf=True, sublinear_tf=False)
        matrix = tfidf.fit_transform(count_matrix).tocsr()
        document_frequency = np.asarray((count_matrix > 0).sum(axis=0)).ravel().astype(int)

        self._counts = counts
        self._tfidf = tfidf
        self.document_ids_ = document_ids
        self.n_documents_ = len(corpus)
        self._document_frequency = document_frequency
        return matrix

    def transform(self, texts: Iterable[str]) -> csr_matrix:
        """Vectorise des textes avec les statistiques apprises, sans réapprentissage.

        Les mots hors vocabulaire sont ignorés. Une entrée vide ou sans terme
        connu produit une ligne nulle. Une liste vide produit une matrice à zéro
        ligne, avec le même nombre de colonnes que le corpus.
        """
        self._check_fitted()
        if isinstance(texts, str):
            raise TypeError("Fournir une liste de textes ou utiliser vectorize_query")
        frequencies = [
            Counter(preprocess_text(text, self._config, stopwords=self._stopwords))
            for text in texts
        ]
        if not frequencies:
            return csr_matrix((0, len(self._counts.vocabulary_)), dtype=float)
        counts = self._counts.transform(frequencies)
        return self._tfidf.transform(counts).tocsr()

    def vectorize_query(self, query: str) -> csr_matrix:
        """Renvoie le vecteur CSR (1, taille du vocabulaire) d'une requête."""
        return self.transform([query])

    def save(self, path: str | Path) -> None:
        """Sauvegarde vocabulaire, IDF, df, identifiants et prétraitement via joblib."""
        self._check_fitted()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"version": 1, "model": self}, target)

    @classmethod
    def load(cls, path: str | Path) -> "TfidfModel":
        """Recharge un modèle sauvegardé dans un environnement compatible.

        Charger uniquement un fichier de confiance : joblib utilise pickle,
        dont la désérialisation peut exécuter du code.
        """
        payload = joblib.load(path)
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError("Format de sauvegarde TF-IDF non reconnu")
        model = payload.get("model")
        if not isinstance(model, cls):
            raise ValueError("La sauvegarde ne contient pas de modèle TF-IDF")
        model._check_fitted()
        return model

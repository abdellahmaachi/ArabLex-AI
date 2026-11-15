"""Recherche lexicale : orientation par Chi², puis classement TF-IDF/cosinus."""

import json
import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import TypedDict

import joblib
from scipy.sparse import csr_matrix, save_npz
from sklearn.exceptions import NotFittedError

from src.common.document_schema import Document
from src.lexical_search.chi2_selector import compute_chi2_scores, select_discriminant_terms
from src.lexical_search.cosine_search import rank_top_k, search_top_k
from src.lexical_search.indexer import InvertedIndex, build_inverted_index
from src.lexical_search.preprocessing import (
    DEFAULT_STOPWORDS,
    PreprocessingConfig,
    preprocess_text,
)
from src.lexical_search.tfidf import TfidfModel


class LexicalHit(Document):
    """Copie du document original enrichie de sa similarité cosinus."""

    score: float


class LexicalSearchResponse(TypedDict):
    query: str
    tokens: list[str]
    selected_class: str | None
    class_scores: dict[str, float]
    results: list[LexicalHit]


@dataclass
class _SearchState:
    documents: dict[str, Document]
    rows_by_class: dict[str, list[int]]
    chi2_scores: dict[str, dict[str, float]]
    selected_terms: dict[str, list[str]]
    model: TfidfModel
    matrix: csr_matrix
    inverted_index: InvertedIndex


class LexicalSearchService:
    """Prépare le corpus une fois, puis recherche sans réapprendre les statistiques.

    Chi² sert à orienter la requête vers une classe, pas à supprimer des colonnes
    TF-IDF. Parmi les termes associés positivement à chaque classe, on conserve
    les ``chi2_k`` meilleurs scores dépassant ``chi2_min_score``. Le score d'une
    classe est la somme des Chi² de ses termes sélectionnés présents dans la
    requête (une seule contribution par terme). Ce score n'est pas une probabilité.

    Sans gagnant unique de score positif, la recherche porte sur toutes les classes.
    Le classement utilise les IDF du corpus entier et le vocabulaire complet.
    Une classe peut aussi être imposée explicitement à ``search``.

    Exemple ::

        service = LexicalSearchService().fit(documents)
        response = service.search("عقد ملكية", k=5)
    """

    def __init__(
        self,
        config: PreprocessingConfig | None = None,
        *,
        stopwords: Iterable[str] = DEFAULT_STOPWORDS,
        chi2_k: int | None = 100,
        chi2_min_score: float = 0.0,
    ) -> None:
        # Réutiliser les validations du sélecteur, même avant l'apprentissage.
        select_discriminant_terms({}, k=chi2_k, min_score=chi2_min_score)
        self._config = config if config is not None else PreprocessingConfig()
        self._stopwords = tuple(stopwords)
        self._chi2_k = chi2_k
        self._chi2_min_score = chi2_min_score
        self._state: _SearchState | None = None

    def _require_state(self) -> _SearchState:
        if self._state is None:
            raise NotFittedError("Appeler fit avant de lancer une recherche lexicale")
        return self._state

    @property
    def selected_terms_(self) -> dict[str, list[str]]:
        """Copie des termes utilisés pour orienter les requêtes."""
        return {label: list(terms) for label, terms in self._require_state().selected_terms.items()}

    @property
    def chi2_scores_(self) -> dict[str, dict[str, float]]:
        """Copie des scores Chi² complets, y compris les associations négatives."""
        return {label: dict(scores) for label, scores in self._require_state().chi2_scores.items()}

    def fit(self, documents: Iterable[Document]) -> "LexicalSearchService":
        """Prépare Chi² et TF-IDF à partir d'au moins deux classes non vides.

        Les documents sont copiés pour préserver leur texte et figer l'alignement
        avec la matrice. Un apprentissage échoué conserve l'état précédent.
        """
        corpus = [document.copy() for document in documents]
        scores = compute_chi2_scores(corpus, self._config, stopwords=self._stopwords)
        index = build_inverted_index(corpus, self._config, stopwords=self._stopwords)
        labels = {document["id"]: document["class"] for document in corpus}
        class_counts = Counter(labels.values())
        positive_scores: dict[str, dict[str, float]] = {label: {} for label in scores}
        for term, postings in index.items():
            present = Counter(labels[doc_id] for doc_id in postings)
            for label in scores:
                # P(terme | classe) > P(terme | autres classes), sans division.
                if present[label] * len(corpus) > class_counts[label] * len(postings):
                    positive_scores[label][term] = scores[label][term]
        selected = select_discriminant_terms(
            positive_scores, k=self._chi2_k, min_score=self._chi2_min_score
        )

        model = TfidfModel(self._config, stopwords=self._stopwords)
        matrix = model.fit_transform(corpus)
        rows_by_class: dict[str, list[int]] = {label: [] for label in scores}
        for row, document in enumerate(corpus):
            rows_by_class[document["class"]].append(row)
        self._state = _SearchState(
            documents={document["id"]: document for document in corpus},
            rows_by_class=rows_by_class,
            chi2_scores=scores,
            selected_terms=selected,
            model=model,
            matrix=matrix,
            inverted_index=index,
        )
        return self

    def save_artifacts(self, directory: str | Path) -> None:
        """Exporte tous les artefacts dans un nouveau dossier (doit être absent).

        ``service.joblib`` permet de reprendre la recherche sans apprentissage.
        Les fichiers séparés permettent d'inspecter et réutiliser chaque étape.
        Le manifeste est écrit en dernier. Un dossier incomplet peut subsister
        si l'écriture échoue ; il ne doit pas être publié comme index actif.
        """
        state = self._require_state()
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=False)
        json_artifacts = {
            "documents.json": [state.documents[doc_id] for doc_id in state.model.document_ids_],
            "inverted_index.json": state.inverted_index,
            "chi2_scores.json": state.chi2_scores,
            "selected_terms.json": state.selected_terms,
            "tfidf_statistics.json": {
                "document_ids": state.model.document_ids_,
                "n_documents": state.model.n_documents_,
                "vocabulary": state.model.vocabulary_,
                "idf": state.model.idf_.tolist(),
                "document_frequency": state.model.document_frequency_.tolist(),
            },
        }
        for filename, data in json_artifacts.items():
            (target / filename).write_text(
                json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
            )
        state.model.save(target / "tfidf_model.joblib")
        save_npz(target / "tfidf_matrix.npz", state.matrix)
        joblib.dump({"version": 1, "service": self}, target / "service.joblib")
        manifest = {
            "version": 1,
            "n_documents": state.model.n_documents_,
            "class_counts": {label: len(rows) for label, rows in state.rows_by_class.items()},
            "matrix_shape": list(state.matrix.shape),
            "matrix_nnz": state.matrix.nnz,
            "preprocessing": asdict(self._config),
            "stopwords": sorted(self._stopwords),
            "chi2_k": self._chi2_k,
            "chi2_min_score": self._chi2_min_score,
            "packages": {
                name: version(name) for name in ("numpy", "scipy", "scikit-learn", "nltk", "joblib")
            },
            "files": [*json_artifacts, "tfidf_model.joblib", "tfidf_matrix.npz", "service.joblib"],
        }
        (target / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "LexicalSearchService":
        """Recharge un fichier service.joblib de confiance, sans réapprentissage.

        Joblib utilise pickle : ne jamais charger un fichier de provenance inconnue.
        Utiliser les versions de bibliothèques indiquées dans le manifeste.
        """
        payload = joblib.load(path)
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError("Format de sauvegarde du service lexical non reconnu")
        service = payload.get("service")
        if not isinstance(service, cls):
            raise ValueError("La sauvegarde ne contient pas de service lexical")
        service._require_state()
        return service

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        min_score: float = 0.0,
        class_name: str | None = None,
    ) -> LexicalSearchResponse:
        """Renvoie classe choisie, scores de classe et Top-K avec textes originaux.

        Les égalités de scores de classe (à la précision flottante près) déclenchent
        une recherche globale. Les résultats cosinus sont départagés par identifiant.
        Une requête vide ou hors vocabulaire ne produit aucun résultat au seuil
        par défaut. ``class_name`` doit désigner une classe présente dans le corpus.
        """
        state = self._require_state()
        rank_top_k([], [], k=k, min_score=min_score)
        if class_name is not None and class_name not in state.rows_by_class:
            raise ValueError(f"Classe inconnue : {class_name}")
        tokens = preprocess_text(query, self._config, stopwords=self._stopwords)
        query_terms = set(tokens)
        class_scores = {
            label: sum(
                (state.chi2_scores[label][term] for term in terms if term in query_terms),
                0.0,
            )
            for label, terms in state.selected_terms.items()
        }
        selected_class = class_name
        if selected_class is None:
            best_score = max(class_scores.values())
            winners = [
                label for label, score in class_scores.items()
                if math.isclose(score, best_score, rel_tol=1e-12, abs_tol=1e-12)
            ]
            if best_score > 0 and len(winners) == 1:
                selected_class = winners[0]

        rows = (
            state.rows_by_class[selected_class]
            if selected_class is not None else list(range(state.matrix.shape[0]))
        )
        # Vectoriser le texte brut : les tokens racinisés ne doivent pas l'être à nouveau.
        query_vector = state.model.vectorize_query(query)
        hits = search_top_k(
            query_vector,
            state.matrix[rows],
            (state.model.document_ids_[row] for row in rows),
            k=k,
            min_score=min_score,
        )
        results: list[LexicalHit] = [
            {**state.documents[hit["id"]], "score": hit["score"]} for hit in hits
        ]
        return {
            "query": query,
            "tokens": tokens,
            "selected_class": selected_class,
            "class_scores": class_scores,
            "results": results,
        }

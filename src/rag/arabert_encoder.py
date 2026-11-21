"""Embeddings AraBERT locaux : mean pooling masqué et normalisation L2."""

from collections.abc import Iterable
from pathlib import Path

import numpy as np


class AraBERTEncoder:
    """Encode documents et requêtes dans le même espace de dimension hidden_size.

    ``model_dir`` contient un checkpoint Transformers AraBERT/BERT et son
    tokenizer, idéalement adaptés à la similarité sémantique. Le pooling seul
    ne constitue pas un entraînement contrastif. Aucun téléchargement ni
    prétraitement arabe externe n'est effectué : fournir le texte selon les
    conventions du checkpoint (notamment la segmentation éventuelle).

    Le pooling moyen exclut padding et tokens spéciaux. Les vecteurs float32
    sont normalisés L2, utilisables avec cosinus ou produit scalaire. Les textes
    blancs et les représentations nulles/non finies sont refusés.

    CPU par défaut. PyTorch et Transformers sont importés à l'instanciation.
    Les textes dépassant ``max_length`` sont tronqués par défaut ; cette limite
    compte les sous-tokens, tokens spéciaux inclus, contrairement au chunker.
    Avec ``truncate=False``, ils sont refusés avant l'inférence du lot concerné.
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int | None = None,
        truncate: bool = True,
    ) -> None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size doit être un entier strictement positif")
        if type(truncate) is not bool:
            raise TypeError("truncate doit être un booléen")
        path = Path(model_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"Dossier du modèle introuvable : {path}")
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Installer PyTorch et les dépendances NLP : pip install -e '.[nlp]'"
            ) from exc

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False,
        )
        # La tête pooler BERT n'est pas utilisée pour la moyenne des tokens.
        self.model, loading_info = AutoModel.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False,
            add_pooling_layer=False, output_loading_info=True,
        )
        if loading_info.get("missing_keys") or loading_info.get("mismatched_keys"):
            raise ValueError("Checkpoint incomplet : certains poids ne sont pas chargés")
        if self.model.config.model_type != "bert":
            raise ValueError("Un modèle AraBERT/BERT est requis")
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Le tokenizer doit définir un token de padding")
        limit = min(self.model.config.max_position_embeddings, self.tokenizer.model_max_length)
        minimum = self.tokenizer.num_special_tokens_to_add(pair=False) + 1
        self.max_length = limit if max_length is None else max_length
        if type(self.max_length) is not int or not minimum <= self.max_length <= limit:
            raise ValueError(f"max_length doit être un entier entre {minimum} et {limit}")
        self.dimension = self.model.config.hidden_size
        self.batch_size = batch_size
        self.truncate = truncate
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        """Renvoie une matrice float32 (nombre de textes, dimension), dans l'ordre.

        Une collection vide donne une matrice (0, dimension). Utiliser
        ``encode_query`` pour une chaîne unique. Les textes doivent déjà être
        préparés de manière compatible avec le modèle.
        """
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts doit être une collection de chaînes, pas une chaîne")
        values = list(texts)
        for index, text in enumerate(values):
            if not isinstance(text, str):
                raise TypeError(f"texts[{index}] doit être une chaîne")
            if not text.strip():
                raise ValueError(f"texts[{index}] est vide")
        result = np.empty((len(values), self.dimension), dtype=np.float32)
        self.model.eval()
        torch = self._torch
        with torch.inference_mode():
            for start in range(0, len(values), self.batch_size):
                batch = values[start:start + self.batch_size]
                encoded = self.tokenizer(
                    batch, padding=True, truncation=self.truncate,
                    max_length=self.max_length if self.truncate else None,
                    return_tensors="pt", return_attention_mask=True,
                    return_special_tokens_mask=True,
                )
                if encoded["input_ids"].shape[1] > self.max_length:
                    raise ValueError("Texte trop long : segmenter ou activer truncate")
                special = encoded.pop("special_tokens_mask").to(self.device)
                inputs = {key: value.to(self.device) for key, value in encoded.items()}
                mask = inputs["attention_mask"].bool() & ~special.bool()
                counts = mask.sum(dim=1, keepdim=True)
                if bool((counts == 0).any()):
                    raise ValueError("Un texte ne contient aucun token de contenu")
                hidden = self.model(**inputs, return_dict=True).last_hidden_state.float()
                if tuple(hidden.shape) != (*mask.shape, self.dimension):
                    raise ValueError("Dimensions des représentations incompatibles")
                pooled = hidden.masked_fill(~mask.unsqueeze(-1), 0).sum(dim=1) / counts
                norms = torch.linalg.vector_norm(pooled, dim=1, keepdim=True)
                if not bool(torch.isfinite(pooled).all()) or not bool(torch.isfinite(norms).all()):
                    raise ValueError("Le modèle a produit des embeddings non finis")
                if bool((norms <= 1e-12).any()):
                    raise ValueError("Le modèle a produit un embedding nul")
                result[start:start + len(batch)] = (pooled / norms).cpu().numpy()
        return result

    def encode_documents(self, texts: Iterable[str]) -> np.ndarray:
        """Encode les textes des passages, par exemple ``p['text'] for p in passages``."""
        return self.encode(texts)

    def encode_query(self, text: str) -> np.ndarray:
        """Renvoie un vecteur float32 (dimension,) avec le même traitement que le corpus."""
        return self.encode([text])[0]

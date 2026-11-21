"""Inférence locale AraBERT NER avec offsets dans le texte arabe original."""

import json
from pathlib import Path

from src.ner.bio_decoder import Entity, decode_bio, validate_bio_labels


class AraBERTAnnotator:
    """Charge un dossier ``best_model`` exporté par le notebook NER.

    Exemple ::

        annotator = AraBERTAnnotator("artifacts/ner/mon_run/best_model")
        entities = annotator.annotate("حكمت محكمة النقض في القضية.")

    Les résultats contiennent ``text``, ``label``, ``start`` et ``end``.
    Les offsets sont des indices Python, fin exclusive : ``text[start:end]``.
    Aucun prétraitement externe, téléchargement ou entraînement n'est effectué.

    Pour les textes longs, les fenêtres chevauchent ``stride`` sous-tokens.
    On conserve pour chaque sous-token la prédiction la plus éloignée du bord
    d'une fenêtre (première fenêtre en cas d'égalité), puis décode globalement.
    Les frontières réelles étant inconnues à l'inférence, cette stratégie ne
    garantit pas la justesse des entités traversant une frontière de fenêtre.

    ``invalid_bio='ignore'`` ignore les I-TYPE sans B/I du même type, comme
    l'évaluation IOB2 stricte ; ``'raise'`` refuse ces transitions. Aucun I
    isolé n'est transformé en B. CPU par défaut ; ``device='cuda'`` est permis.
    PyTorch et Transformers sont importés seulement à l'instanciation.
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str = "cpu",
        max_length: int | None = None,
        stride: int | None = None,
        invalid_bio: str = "ignore",
    ) -> None:
        path = Path(model_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"Dossier du modèle introuvable : {path}")
        if invalid_bio not in {"ignore", "raise"}:
            raise ValueError("invalid_bio doit être 'ignore' ou 'raise'")
        metadata_path = path / "ner_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        if not isinstance(metadata, dict):
            raise ValueError("ner_metadata.json doit contenir un objet")
        if metadata.get("preprocessing", "original_text_no_external_normalization") != (
            "original_text_no_external_normalization"
        ):
            raise ValueError("Le prétraitement du modèle n'est pas compatible avec les offsets")
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Installer PyTorch et les dépendances NLP : pip install -e '.[nlp]'") from exc

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(path), use_fast=True, local_files_only=True, trust_remote_code=False,
        )
        if not self.tokenizer.is_fast:
            raise ValueError("Un tokenizer rapide est requis pour les offsets")
        if self.tokenizer.cls_token_id is None or self.tokenizer.sep_token_id is None:
            raise ValueError("Le tokenizer doit définir [CLS] et [SEP]")
        self.model, loading_info = AutoModelForTokenClassification.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False, output_loading_info=True,
        )
        if loading_info.get("missing_keys") or loading_info.get("mismatched_keys"):
            raise ValueError("Checkpoint incomplet : certains poids NER ne sont pas chargés")
        config = self.model.config
        if config.model_type != "bert":
            raise ValueError("Un modèle AraBERT/BERT est requis")
        self.id2label = dict(config.id2label)
        if set(self.id2label) != set(range(config.num_labels)):
            raise ValueError("id2label doit couvrir les indices de sortie du modèle")
        labels = list(self.id2label.values())
        if any(not isinstance(label, str) for label in labels) or len(set(labels)) != len(labels):
            raise ValueError("Labels BIO non textuels ou dupliqués")
        if "O" not in labels:
            raise ValueError("Le modèle doit définir le label O")
        for label in labels:
            validate_bio_labels([f"B-{label[2:]}", label] if label.startswith("I-") else [label])
            if label != "O" and any(prefix + label[2:] not in labels for prefix in ("B-", "I-")):
                raise ValueError(f"Paire BIO incomplète : {label}")
        if config.label2id != {label: index for index, label in self.id2label.items()}:
            raise ValueError("Mappings label2id/id2label incohérents")
        if "label2id" in metadata and metadata["label2id"] != config.label2id:
            raise ValueError("Labels des métadonnées différents de ceux du modèle")
        limit = min(config.max_position_embeddings, self.tokenizer.model_max_length)
        self.max_length = max_length if max_length is not None else metadata.get("max_length", limit)
        if type(self.max_length) is not int or not 3 <= self.max_length <= limit:
            raise ValueError(f"max_length doit être un entier entre 3 et {limit}")
        capacity = self.max_length - 2
        self.stride = min(64, capacity // 2) if stride is None else stride
        if type(self.stride) is not int or not 0 <= self.stride < capacity:
            raise ValueError("stride doit être un entier entre 0 et max_length - 3")
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.invalid_bio = invalid_bio

    def annotate(self, text: str) -> list[Entity]:
        """Extrait les entités sans tronquer le texte ni modifier ses caractères."""
        if not isinstance(text, str):
            raise TypeError("text doit être une chaîne")
        if not text.strip():
            return []
        encoded = self.tokenizer(
            text, add_special_tokens=False, truncation=False,
            return_offsets_mapping=True, return_attention_mask=False,
            return_token_type_ids=False,
        )
        token_ids = encoded["input_ids"]
        offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
        if len(token_ids) != len(offsets):
            raise ValueError("Nombre d'offsets différent du nombre de sous-tokens")
        # Valide aussi les offsets des tokens prédits O.
        decode_bio(text, ["O"] * len(token_ids), offsets)
        if not token_ids:
            return []
        predictions = ["O"] * len(token_ids)
        context_sizes = [-1] * len(token_ids)
        capacity = self.max_length - 2
        start = 0
        self.model.eval()
        with self._torch.inference_mode():
            while start < len(token_ids):
                end = min(start + capacity, len(token_ids))
                ids = [self.tokenizer.cls_token_id, *token_ids[start:end], self.tokenizer.sep_token_id]
                inputs = {
                    "input_ids": self._torch.tensor([ids], dtype=self._torch.long, device=self.device),
                    "attention_mask": self._torch.ones((1, len(ids)), dtype=self._torch.long, device=self.device),
                }
                if "token_type_ids" in self.tokenizer.model_input_names:
                    inputs["token_type_ids"] = self._torch.zeros_like(inputs["input_ids"])
                logits = self.model(**inputs).logits
                if tuple(logits.shape) != (1, len(ids), len(self.id2label)):
                    raise ValueError("Dimensions des logits incompatibles avec les labels")
                if not bool(self._torch.isfinite(logits).all()):
                    raise ValueError("Le modèle a produit des logits non finis")
                # Ne jamais décoder les positions [CLS] et [SEP].
                selected = logits[0, 1:-1].argmax(dim=-1).tolist()
                for local_index, label_id in enumerate(selected):
                    position = start + local_index
                    context = min(local_index, end - start - 1 - local_index)
                    if context > context_sizes[position]:
                        predictions[position] = self.id2label[label_id]
                        context_sizes[position] = context
                if end == len(token_ids):
                    break
                start = end - self.stride

        if self.invalid_bio == "ignore":
            active_type = None
            for index, label in enumerate(predictions):
                if label.startswith("I-") and active_type != label[2:]:
                    predictions[index] = "O"
                active_type = None if predictions[index] == "O" else label[2:]
        return decode_bio(text, predictions, offsets)

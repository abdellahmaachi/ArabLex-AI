"""Configuration sans chargement de modèle ni création de dossier.

Priorité : arguments > environnement > .env à la racine > valeurs par défaut.
Les variables d'environnement portent le nom des attributs en majuscules.
Les chemins relatifs sont ancrés à la racine du dépôt, indépendamment du cwd.
Settings(_env_file=None) désactive .env ; get_settings.cache_clear() vide le cache.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Options du backend ; les artefacts peuvent ne pas encore exister.

    Les modèles de classification et NER doivent être choisis explicitement.
    Les services liront les pointeurs current.json publiés par les scripts.
    Gemini reste facultatif pour permettre l'utilisation des autres services.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8",
        extra="ignore", case_sensitive=False, frozen=True,
        hide_input_in_errors=True, allow_inf_nan=False,
    )

    app_env: str = Field(default="development", min_length=1)
    log_level: str = "INFO"
    cors_origins: Annotated[tuple[str, ...], NoDecode] = ("http://localhost:4200",)

    corpus_dir: Path = PROJECT_ROOT / "data" / "raw"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    classification_model_path: Path | None = None
    lexical_index_path: Path | None = None
    ner_model_dir: Path | None = None
    ner_index_path: Path | None = None
    rag_index_path: Path | None = None
    rag_encoder_dir: Path | None = None
    qdrant_path: Path | None = None
    qdrant_collection: str = Field(default="legal_passages", min_length=1)

    inference_device: str = Field(default="cpu", min_length=1)
    lexical_top_k: int = Field(default=10, gt=0)
    rag_top_k: int = Field(default=5, gt=0)
    rag_batch_size: int = Field(default=32, gt=0)
    gemini_api_key: SecretStr | None = Field(default=None, repr=False)
    gemini_model: str | None = None
    gemini_timeout_ms: int = Field(default=30_000, gt=0)
    gemini_max_attempts: int = Field(default=3, gt=0)
    gemini_max_output_tokens: int = Field(default=2048, gt=0)
    gemini_temperature: float = Field(default=0.2, ge=0, le=2)

    @field_validator("gemini_api_key", "gemini_model", mode="before")
    @classmethod
    def empty_gemini_value(cls, value):
        return (value.strip() or None) if isinstance(value, str) else value

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        value = value.strip().upper()
        if value not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}:
            raise ValueError("Niveau de journalisation inconnu")
        return value

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value):
        if isinstance(value, str):
            value = json.loads(value) if value.lstrip().startswith("[") else value.split(",")
        if isinstance(value, (list, tuple)):
            if any(not isinstance(origin, str) for origin in value):
                raise ValueError("Les origines CORS doivent être des chaînes")
            return tuple(origin.strip() for origin in value if origin.strip())
        return value

    @model_validator(mode="after")
    def resolve_paths(self) -> Self:
        for name in ("corpus_dir", "artifacts_dir", "classification_model_path",
                     "lexical_index_path", "ner_model_dir", "ner_index_path",
                     "rag_index_path", "rag_encoder_dir", "qdrant_path"):
            value = getattr(self, name)
            if value is not None:
                value = value.expanduser()
                if not value.is_absolute():
                    value = PROJECT_ROOT / value
                object.__setattr__(self, name, value.resolve())
        defaults = {
            "lexical_index_path": "lexical_search/current.json",
            "ner_index_path": "ner/entity_index/index.json",
            "rag_index_path": "rag/current.json",
            "rag_encoder_dir": "rag/encoder",
            "qdrant_path": "rag/qdrant",
        }
        for name, relative in defaults.items():
            if getattr(self, name) is None:
                object.__setattr__(self, name, self.artifacts_dir / relative)
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Dépendance FastAPI : lecture différée et mise en cache des paramètres."""
    return Settings()

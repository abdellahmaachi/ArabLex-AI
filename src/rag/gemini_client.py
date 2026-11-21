"""Client synchrone Gemini Flash pour les prompts RAG, via google-genai."""

import importlib
import math
import os
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import errors, types

_RETRY_CODES = (408, 429, 500, 502, 503, 504)


def _transport_errors() -> tuple[type[Exception], ...]:
    # Le SDK prend en charge httpx ou son fork httpx2 selon l'installation.
    exceptions: list[type[Exception]] = []
    for name in ("httpx", "httpx2"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        exceptions.append(module.TransportError)
    return tuple(exceptions)


_TRANSPORT_ERRORS = _transport_errors()


class GeminiError(RuntimeError):
    """Erreur applicative Gemini, sans prompt ni clé dans son message."""


class GeminiRequestError(GeminiError):
    """Échec API ou réseau après les tentatives gérées par le SDK."""

    def __init__(self, message: str, *, status_code: int | None, retryable: bool) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class GeminiResponseError(GeminiError):
    """Réponse absente, vide ou interrompue pour une raison inattendue."""

    def __init__(self, message: str, *, reason: str | None = None) -> None:
        super().__init__(message)
        self.reason = reason


class GeminiBlockedError(GeminiResponseError):
    """Prompt ou réponse bloqué par le service."""


class GeminiTruncatedError(GeminiResponseError):
    """Réponse arrêtée par la limite de tokens ; aucun texte partiel n'est livré."""


@dataclass(frozen=True)
class GeminiResponse:
    """Réponse textuelle complète et compteurs de tokens fournis par le service."""

    text: str
    model: str
    finish_reason: str
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


def _reason(value: Any) -> str | None:
    return getattr(value, "value", value) if value is not None else None


class GeminiClient:
    """Appelle un modèle Flash configuré explicitement ou par GEMINI_MODEL.

    La clé vient de ``api_key`` ou de ``GEMINI_API_KEY``. Le fichier .env n'est
    pas chargé automatiquement : l'application configure son environnement.
    Aucun nom de modèle n'est figé pour éviter un modèle retiré du service.

    ``timeout_ms`` borne chaque tentative HTTP, pas la durée globale ;
    ``max_attempts`` inclut la première. Le SDK applique le backoff sur les
    erreurs transitoires, sans boucle de retry supplémentaire ici. Les refus,
    erreurs d'authentification et réponses incomplètes ne sont pas réessayés.
    Aucun prompt, texte de réponse ou secret n'est journalisé par ce module.

    Exemple : ``with GeminiClient() as client: answer = client.generate_text(prompt)``.
    """

    def __init__(
        self, *, api_key: str | None = None, model: str | None = None,
        timeout_ms: int = 30_000, max_attempts: int = 3,
        max_output_tokens: int = 2048, temperature: float = 0.2,
    ) -> None:
        key = os.getenv("GEMINI_API_KEY") if api_key is None else api_key
        selected_model = os.getenv("GEMINI_MODEL") if model is None else model
        for name, value in (("GEMINI_API_KEY", key), ("GEMINI_MODEL", selected_model)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} doit être configuré et non vide")
        for name, value in (("timeout_ms", timeout_ms), ("max_attempts", max_attempts),
                            ("max_output_tokens", max_output_tokens)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} doit être un entier strictement positif")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise TypeError("temperature doit être un nombre réel")
        if not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ValueError("temperature doit être comprise entre 0 et 2")
        self.model = selected_model.strip()
        self._config = types.GenerateContentConfig(
            temperature=temperature, max_output_tokens=max_output_tokens,
            candidate_count=1, response_mime_type="text/plain",
        )
        self._client = genai.Client(
            api_key=key.strip(), vertexai=False,
            http_options=types.HttpOptions(
                timeout=timeout_ms,
                retry_options=types.HttpRetryOptions(
                    attempts=max_attempts, initial_delay=1, max_delay=10,
                    exp_base=2, jitter=1, http_status_codes=list(_RETRY_CODES),
                ),
            ),
        )
        self._closed = False

    def generate(self, prompt: str) -> GeminiResponse:
        """Génère une réponse complète ou lève une exception GeminiError typée.

        Seul un candidat terminé par STOP est accepté. Les parties de réflexion
        (thought) sont exclues du texte rendu. Les erreurs API sont traduites sans
        reprendre leurs détails, susceptibles de contenir des données de requête.
        Les compteurs peuvent être absents ; total_tokens peut inclure la réflexion.
        """
        if self._closed:
            raise RuntimeError("Le client Gemini est fermé")
        if not isinstance(prompt, str):
            raise TypeError("prompt doit être une chaîne")
        if not prompt.strip():
            raise ValueError("prompt doit être non vide")
        try:
            response = self._client.models.generate_content(
                model=self.model, contents=prompt, config=self._config,
            )
        except errors.APIError as exc:
            code = exc.code
            raise GeminiRequestError(
                f"Échec de l'appel Gemini (HTTP {code}).",
                status_code=code, retryable=code in _RETRY_CODES,
            ) from None
        except _TRANSPORT_ERRORS:
            raise GeminiRequestError(
                "Échec réseau ou délai d'attente Gemini dépassé.",
                status_code=None, retryable=True,
            ) from None
        feedback = getattr(response, "prompt_feedback", None)
        blocked = _reason(getattr(feedback, "block_reason", None))
        if blocked and blocked != "BLOCKED_REASON_UNSPECIFIED":
            raise GeminiBlockedError("Le service Gemini a bloqué le prompt.", reason=blocked)
        candidates = getattr(response, "candidates", None)
        if not candidates or len(candidates) != 1:
            raise GeminiResponseError("Gemini n'a pas renvoyé un candidat unique.")
        candidate = candidates[0]
        reason = _reason(candidate.finish_reason)
        if reason == "MAX_TOKENS":
            raise GeminiTruncatedError("La réponse Gemini dépasse la limite de tokens.", reason=reason)
        if reason in {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"}:
            raise GeminiBlockedError("Le service Gemini a bloqué la réponse.", reason=reason)
        if reason != "STOP":
            raise GeminiResponseError("La génération Gemini ne s'est pas terminée normalement.", reason=reason)
        parts = getattr(candidate.content, "parts", None) or []
        text = "".join(part.text for part in parts if part.text and not part.thought).strip()
        if not text:
            raise GeminiResponseError("Gemini a renvoyé une réponse textuelle vide.", reason=reason)
        usage = getattr(response, "usage_metadata", None)
        return GeminiResponse(
            text=text, model=getattr(response, "model_version", None) or self.model,
            finish_reason=reason,
            prompt_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
            total_tokens=getattr(usage, "total_token_count", None),
        )

    def generate_text(self, prompt: str) -> str:
        """Renvoie uniquement le texte, avec les mêmes contrôles que generate."""
        return self.generate(prompt).text

    def close(self) -> None:
        """Ferme les connexions HTTP ; plusieurs appels sont permis."""
        if not self._closed:
            self._client.close()
            self._closed = True

    def __enter__(self) -> "GeminiClient":
        if self._closed:
            raise RuntimeError("Le client Gemini est fermé")
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

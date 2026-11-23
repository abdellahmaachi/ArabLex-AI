"""Schémas publics des quatre services de l'API."""

from .classification import ClassificationRequest, ClassificationResponse
from .lexical import LexicalHit, LexicalSearchRequest, LexicalSearchResponse
from .ner import (
    Entity,
    EntitySearchHit,
    EntitySearchRequest,
    EntitySearchResponse,
    NERRequest,
    NERResponse,
)
from .rag import GenerationInfo, Passage, RAGRequest, RAGResponse, RAGSource

__all__ = [
    "ClassificationRequest", "ClassificationResponse",
    "LexicalHit", "LexicalSearchRequest", "LexicalSearchResponse",
    "Entity", "EntitySearchHit", "EntitySearchRequest", "EntitySearchResponse",
    "NERRequest", "NERResponse",
    "GenerationInfo", "Passage", "RAGRequest", "RAGResponse", "RAGSource",
]

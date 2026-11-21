"""Orchestration RAG : récupération, prompt, génération et références sources."""

import re
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Protocol, TypedDict

from src.rag.prompt_builder import build_prompt

if TYPE_CHECKING:
    from src.rag.gemini_client import GeminiResponse

_CITATION = re.compile(r"\[(S[0-9]+)\]")
_ARABIC = re.compile(r"[\u0600-\u06ff]")


class Retriever(Protocol):
    def retrieve(
        self, query: str, *, top_k: int = 5,
        document_id: str | None = None, class_name: str | None = None,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]: ...


class Generator(Protocol):
    def generate(self, prompt: str) -> "GeminiResponse": ...


class RAGResponse(TypedDict):
    question: str
    answer: str
    sources: list[dict[str, Any]]
    cited_references: list[str]
    citation_status: str
    generation: dict[str, Any] | None


class InvalidCitationError(ValueError):
    """La réponse cite une référence absente des passages envoyés au modèle."""


class RAGService:
    """Assemble RAGRetriever et GeminiClient déjà initialisés.

    L'appelant conserve la responsabilité de fermer les clients. Le service ne
    charge aucun modèle, ne modifie pas l'index et ne journalise pas les prompts.
    Les erreurs du retriever et de Gemini sont propagées sans réponse de secours
    qui masquerait un incident technique.

    Exemple ::

        service = RAGService(retriever, gemini_client)
        result = service.answer("ما هي حقوق الملكية؟", top_k=5)
        print(result["answer"])
    """

    def __init__(self, retriever: Retriever, generator: Generator) -> None:
        self.retriever = retriever
        self.generator = generator

    def answer(
        self, question: str, *, top_k: int = 5,
        document_id: str | None = None, class_name: str | None = None,
        score_threshold: float | None = None,
    ) -> RAGResponse:
        """Renvoie la réponse, ses sources et les informations de génération.

        ``sources`` conserve tous les résultats dans l'ordre du prompt, avec
        ``reference`` (S1, S2...), ``id``, ``score``, ``passage`` et ``cited``.
        Les références citées sont uniques, dans l'ordre d'apparition dans la
        réponse. Les objets renvoyés sont indépendants de ceux du retriever.

        ``citation_status`` vaut ``valid`` si les marqueurs [Sn] désignent des
        sources existantes, ``missing`` si aucun n'apparaît (notamment en cas
        d'abstention), et ``not_applicable`` sans passage récupéré. Ce contrôle
        syntaxique ne vérifie ni la justification des affirmations ni leur
        couverture par les citations. Une référence inconnue lève une erreur.

        Sans passage, aucun appel de génération : message fixe en arabe si la
        question contient des caractères arabes, en français sinon. Les compteurs
        de génération valent alors None. Les scores restent des similarités,
        pas des probabilités d'exactitude de la réponse.
        """
        if not isinstance(question, str):
            raise TypeError("question doit être une chaîne")
        if not question.strip():
            raise ValueError("question doit être non vide")
        # RAGRetriever valide les options avant toute inférence.
        hits = deepcopy(self.retriever.retrieve(
            question, top_k=top_k, document_id=document_id,
            class_name=class_name, score_threshold=score_threshold,
        ))
        # Valider les sources et figer le même ordre pour le prompt et la réponse.
        prompt = build_prompt(question, hits)
        if not hits:
            message = (
                "لم يتم العثور على مقاطع ذات صلة في مجموعة الوثائق. لا يمكن الإجابة دون مصادر."
                if _ARABIC.search(question) else
                "Aucun passage pertinent n'a été trouvé dans le corpus. "
                "Les sources disponibles ne permettent pas de répondre."
            )
            return RAGResponse(
                question=question, answer=message, sources=[], cited_references=[],
                citation_status="not_applicable", generation=None,
            )
        generated = self.generator.generate(prompt)
        if not isinstance(generated.text, str) or not generated.text.strip():
            raise ValueError("Le générateur a renvoyé une réponse textuelle vide")
        references = list(dict.fromkeys(_CITATION.findall(generated.text)))
        available = {f"S{index}" for index in range(1, len(hits) + 1)}
        unknown = [reference for reference in references if reference not in available]
        if unknown:
            raise InvalidCitationError("Références inconnues : " + ", ".join(unknown))
        sources = [
            {"reference": f"S{index}", "id": hit["id"], "score": hit["score"],
             "passage": hit["passage"], "cited": f"S{index}" in references}
            for index, hit in enumerate(hits, start=1)
        ]
        return RAGResponse(
            question=question, answer=generated.text, sources=sources,
            cited_references=references, citation_status="valid" if references else "missing",
            generation={
                "model": generated.model, "finish_reason": generated.finish_reason,
                "prompt_tokens": generated.prompt_tokens, "output_tokens": generated.output_tokens,
                "total_tokens": generated.total_tokens,
            },
        )

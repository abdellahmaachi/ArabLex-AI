"""Types communs des contrats JSON, sans dépendance aux modèles NLP."""

from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field


def _non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("Le texte ne doit pas être vide ou uniquement composé d'espaces")
    return value


# Ne pas retirer les espaces : les positions NER portent sur le texte original.
NonBlankText = Annotated[str, AfterValidator(_non_blank)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
Probability = Annotated[float, Field(ge=0, le=1)]
CosineScore = Annotated[float, Field(ge=-1, le=1)]


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, populate_by_name=True)


class Document(APIModel):
    id: NonBlankText
    class_name: NonBlankText = Field(alias="class")
    filename: str
    path: str
    text: str

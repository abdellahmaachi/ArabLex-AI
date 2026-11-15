"""Structure commune des documents du corpus juridique."""

from typing import TypedDict

# La syntaxe fonctionnelle permet de conserver la clé « class », mot réservé Python.
Document = TypedDict(
    "Document",
    {
        "id": str,
        "class": str,
        "filename": str,
        "path": str,
        "text": str,
    },
)

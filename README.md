# ArabicLegalTextMining

Structure initiale d'une plateforme de Text Mining juridique arabe.

## Services prévus

1. Recherche lexicale : Chi² + TF-IDF + similarité cosinus
2. Classification : Naive Bayes + validation croisée
3. RAG : AraBERT adapté + Qdrant + Gemini Flash
4. Recherche par entités : AraBERT NER + BIO

## Organisation

- `data/` : corpus, annotations et jeux d'évaluation
- `notebooks/` : expérimentation et entraînement des modèles
- `src/` : modules Python des quatre services
- `artifacts/` : modèles et index sauvegardés
- `backend/` : API FastAPI
- `frontend/angular-app/` : application Angular
- `scripts/` : pipelines hors ligne
- `tests/` : tests automatiques

**État :** arborescence initiale uniquement. Les algorithmes, les endpoints et
l'interface seront développés progressivement. Les notebooks sont des fichiers
valides sans code d'entraînement pour l'instant.

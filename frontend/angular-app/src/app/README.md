# Interface des quatre API

> Les quatre routes affichent les formulaires reliés aux API FastAPI dans le
> layout ArabLex IA. Le backend et ses modèles/index doivent être disponibles pour
> exécuter les analyses.

Chaque service injectable utilise les contrats de `models/api.models.ts` et expose
un Observable typé. Les composants standalone sont chargés par le routeur.
Les formulaires valident les champs, empêchent les soumissions concurrentes et
affichent le chargement, les erreurs FastAPI et les résultats vides.
Les textes et résultats utilisent `dir="auto"` pour l’arabe.

## Démarrage local

Lancer FastAPI sur le port 8000 depuis la racine du dépôt :
`uvicorn backend.app.main:app --host 127.0.0.1 --port 8000`.
Configurer les modèles et index selon `.env.example`.
Depuis `frontend/angular-app`, lancer `npm start` et ouvrir `http://localhost:4200`.
Le proxy de développement transmet `/api/**` à FastAPI, sans configuration CORS.

En production, configurer le serveur web pour transmettre `/api` au backend et
servir `index.html` pour les routes Angular. Le proxy de développement ne fait
pas partie du build. Pour une API sur une autre origine, fournir `API_BASE_URL`
dans `app.config.ts` (URL incluant `/api`) et autoriser cette origine côté backend.

| Écran | Endpoint POST |
| --- | --- |
| `/outils/recherche` | `/api/search/lexical` |
| `/outils/classification` | `/api/classify` |
| `/outils/entites` | `/api/ner/search` |
| `/outils/assistant` | `/api/rag/ask` |

Les routes des outils sont chargées à la demande depuis `components/tools.routes.ts`.
Les anciens chemins restent accessibles par redirection ; les chemins inconnus
affichent une page dédiée.

L’assistant conserve les échanges sur l’écran courant, propose des questions,
affiche les sources par réponse et permet de réessayer un appel échoué.
L’historique est effacé en quittant l’écran ou en démarrant une nouvelle
conversation. L’API n’acceptant pas d’historique, chaque question est envoyée
indépendamment, sans simulation de streaming ni contexte implicite.

Les classes du modèle `family_inheritance` et `real_estate` sont affichées en
arabe via `ClassLabelPipe`. Les identifiants utilisés par FastAPI sont conservés.

Vérification : `npm run build` et `npm test -- --watch=false`.

`vitest-base.config.mts` convertit les chemins absolus des tests découverts par
Angular en chemins relatifs pour Vitest. Cela évite l’erreur « No test files
found » lorsque le dossier parent contient des parenthèses, comme `Master(MLAIM)`.

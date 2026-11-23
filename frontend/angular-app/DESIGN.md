# ArabLex IA — Design de l’interface

Le design system, le layout et l’accueil sont complétés par les quatre
formulaires reliés aux services FastAPI. Les appels sont déclenchés uniquement
à la soumission. Les erreurs du backend sont affichées dans les formulaires.

## Démarrage

Depuis `frontend/angular-app`, exécuter `npm.cmd start`, puis ouvrir
`http://localhost:4200/`. Le serveur FastAPI sur le port 8000 est nécessaire pour les analyses.

## Structure

- `src/styles.scss` : couleurs sémantiques, espacements, rayons, ombres,
  typographie système, boutons, cartes et focus.
- `src/app/app.*` : sidebar, topbar, navigation mobile et pied de page.
- `src/app/pages/home/` : accueil, illustration documentaire, cartes et guide.
- `src/app/components/` : formulaires et résultats des quatre services.
- `src/app/pages/tool-preview/` : ancien aperçu local, hors navigation principale.
- `src/app/shared/icon.ts` : icônes SVG réutilisables.
- `public/logo.svg` : balance de justice et livre ouvert, également utilisés
  comme favicon. Le nom de présentation choisi est « ArabLex IA ».

## Principes visuels

Bleu principal `#245BDC`, texte `#172640`, fond `#F6F8FC`, cartes blanches.
Les variables CSS `--color-*`, `--radius-*`, `--space-*` et `--shadow-card`
permettent de faire évoluer le thème sans dupliquer les valeurs communes.
Les polices et illustrations sont locales : aucun CDN ni service d’images.

## Responsive et accessibilité

Sidebar permanente sur grand écran, menu repliable sous 900 px. Les cartes
passent de quatre à deux puis une colonne. Les liens restent accessibles au
clavier, le focus est visible, le menu expose son état et se ferme avec Échap.
Un lien d’évitement mène au contenu principal. Les textes arabes sont déclarés
avec `lang="ar"` et `dir="rtl"`. Les animations respectent la préférence de
réduction des mouvements. Les exemples sont identifiés comme illustrations.

## Vérifications

`npm.cmd run build` et `npm.cmd test -- --watch=false` sont les commandes
complètes. Une vérification visuelle desktop/mobile reste nécessaire lorsqu’un
navigateur est disponible.

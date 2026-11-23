import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', pathMatch: 'full', title: 'Accueil · ArabLex IA', loadComponent: () => import('./pages/home/home').then(m => m.Home) },
  { path: 'outils', loadChildren: () => import('./components/tools.routes').then(m => m.TOOL_ROUTES) },
  // Preserve bookmarks to the previous service URLs.
  { path: 'lexical', pathMatch: 'full', redirectTo: 'outils/recherche' },
  { path: 'classification', pathMatch: 'full', redirectTo: 'outils/classification' },
  { path: 'entities', pathMatch: 'full', redirectTo: 'outils/entites' },
  { path: 'rag', pathMatch: 'full', redirectTo: 'outils/assistant' },
  { path: '**', title: 'Page introuvable · ArabLex IA', loadComponent: () => import('./pages/not-found/not-found').then(m => m.NotFound) },
];

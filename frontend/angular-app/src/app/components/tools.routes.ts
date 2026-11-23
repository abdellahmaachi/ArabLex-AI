import { Routes } from '@angular/router';

export const TOOL_ROUTES: Routes = [
  { path: '', pathMatch: 'full', redirectTo: 'assistant' },
  { path: 'recherche', title: 'Recherche lexicale · ArabLex IA', loadComponent: () => import('./lexical-search/lexical-search.component').then(m => m.LexicalSearchComponent) },
  { path: 'classification', title: 'Classification · ArabLex IA', loadComponent: () => import('./classification/classification.component').then(m => m.ClassificationComponent) },
  { path: 'entites', title: 'Entités juridiques · ArabLex IA', loadComponent: () => import('./entity-search/entity-search.component').then(m => m.EntitySearchComponent) },
  { path: 'assistant', title: 'Assistant juridique · ArabLex IA', loadComponent: () => import('./rag/rag.component').then(m => m.RAGComponent) },
];

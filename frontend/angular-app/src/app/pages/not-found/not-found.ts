import { Component } from '@angular/core';
import { RouterLink } from '@angular/router';
@Component({
  selector: 'app-not-found',
  imports: [RouterLink],
  template: '<section class="card"><p class="eyebrow">PAGE INTROUVABLE</p><h1>Ce chemin ne mène à aucun outil.</h1><p>Retrouvez votre espace de recherche depuis l’accueil.</p><a class="button button-primary" routerLink="/">Revenir à l’accueil</a></section>',
  styles: 'section { max-width: 700px; margin: 3rem auto; padding: 2.5rem; } p { color: var(--color-muted); }',
})
export class NotFound {}

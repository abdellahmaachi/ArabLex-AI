import { Component, inject } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, RouterLink } from '@angular/router';
import { Icon } from '../../shared/icon';
@Component({
  selector: 'app-tool-preview',
  imports: [RouterLink, Icon, FormsModule],
  template: `
    <a class="back-link" routerLink="/">Accueil / Vos outils</a>
    <section class="preview card">
      <span class="preview-icon"><app-icon [name]="tool.icon" /></span>
      <p class="eyebrow">DÉCOUVRIR L’OUTIL</p>
      <h1>{{ tool.title }}</h1><p class="description">{{ tool.description }}</p>
      <div class="example">
        <label class="example-label" for="tool-text">EXEMPLE DE TEXTE · SAISIE LIBRE</label>
        @if (isSearch) {
          <input id="tool-text" class="example-input" type="search" name="text" dir="auto" [(ngModel)]="text" [attr.aria-label]="tool.title" />
        } @else {
          <textarea id="tool-text" class="example-input" name="text" dir="auto" rows="1" [(ngModel)]="text" [attr.aria-label]="tool.title"></textarea>
        }
      </div>
      <div class="notice"><app-icon name="help" /><p>Cet espace présente le futur parcours d’analyse. Aucune requête n’est envoyée et aucun résultat n’est généré dans cette version découverte.</p></div>
      <a class="button button-primary" routerLink="/">Explorer les autres outils</a>
    </section>
  `,
  styles: `
    :host { display: block; max-width: 900px; margin: auto; }
    .back-link { display: inline-block; font-size: .85rem; color: var(--color-muted); margin-bottom: 1.5rem; text-decoration: none; }
    .preview { padding: clamp(1.5rem, 5vw, 3.5rem); }
    .preview-icon { display: inline-flex; padding: 1rem; color: var(--color-primary); background: var(--color-primary-soft); border-radius: 14px; margin-bottom: 1.5rem; }
    h1 { font-size: clamp(1.8rem, 4vw, 2.5rem); letter-spacing: -.035em; }
    .description { color: var(--color-muted); max-width: 600px; }
    .example { padding: 1.5rem; margin: 2rem 0; border: 1px solid var(--color-border); border-radius: 12px; background: var(--color-canvas); }
    .example-label { font-size: .7rem; color: var(--color-muted); letter-spacing: .08em; }
    .example-input { display: block; width: 100%; min-width: 0; padding: 0; border: 0; border-radius: 0; background: transparent; color: inherit; font: inherit; font-size: 1.5rem; line-height: 2; margin: 1rem 0 0; }
    textarea.example-input { resize: vertical; }
    .notice { display: flex; gap: .7rem; color: var(--color-muted); margin-bottom: 1.5rem; }
    .notice p { font-size: .85rem; margin: 0; }
  `,
})
export class ToolPreview {
  private readonly route = inject(ActivatedRoute);
  readonly tool = this.route.snapshot.data['tool'] as {
    title: string; icon: string; description: string; example: string;
  };
  readonly isSearch = ['lexical', 'entities'].includes(this.route.snapshot.routeConfig?.path ?? '');
  // Initialize once, never reassign the example during change detection or typing.
  text = this.tool.example;
}

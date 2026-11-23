import { Component, input } from '@angular/core';
import { Icon } from './icon';

@Component({
  selector: 'app-tool-header',
  imports: [Icon],
  template: `<header><span class="symbol"><app-icon [name]="icon()" /></span><div><p>ARABLEX IA · ESPACE D’ANALYSE</p><h1>{{ title() }}</h1><div class="description">{{ description() }}</div></div></header>`,
  styles: `header { display: flex; gap: 1rem; align-items: flex-start; margin-bottom: 2rem; } .symbol { display: inline-flex; padding: .9rem; background: var(--color-primary-soft); color: var(--color-primary); border-radius: 14px; } p { color: var(--color-primary); font-size: .7rem; letter-spacing: .1em; font-weight: 700; margin: 0 0 .5rem; } h1 { font-size: clamp(1.6rem, 3vw, 2.1rem); letter-spacing: -.035em; margin: 0 0 .6rem; } .description { color: var(--color-muted); font-size: .95rem; max-width: 650px; } @media(max-width: 540px) { .symbol { display: none; } }`,
})
export class ToolHeader {
  readonly icon = input('search');
  readonly title = input('');
  readonly description = input('');
}

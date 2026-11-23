import { Component, input } from '@angular/core';
@Component({
  selector: 'app-icon',
  template: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.65" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path [attr.d]="paths[name()] || paths['search']" /></svg>`,
  styles: `:host { display: inline-flex; width: 1.4rem; height: 1.4rem; flex-shrink: 0; } svg { width: 100%; height: 100%; }`,
})
export class Icon {
  readonly name = input('search');
  readonly paths: Record<string, string> = {
    plus: 'M12 5v14M5 12h14',
    send: 'm22 2-7 20-4-9-9-4Zm0 0L11 13',
    settings: 'M4 7h4m4 0h8M4 17h10m4 0h2M8 4v6m6 4v6',
    home: 'M3 10 12 3l9 7v10a1 1 0 0 1-1 1h-5v-7H9v7H4a1 1 0 0 1-1-1Z',
    search: 'M21 21l-5-5M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0',
    layers: 'm12 3 10 5-10 5L2 8Zm-10 9 10 5 10-5M2 16l10 5 10-5',
    entities: 'M8 4h8v5H8ZM3 16h6v5H3Zm12 0h6v5h-6ZM12 9v4M6 16v-3h12v3',
    spark: 'm12 3 2.4 6.6L21 12l-6.6 2.4L12 21l-2.4-6.6L3 12l6.6-2.4ZM20 2v4m-2-2h4',
    book: 'M12 5v16M12 5C8 2 4 3 2 4v15c4-2 7-1 10 2 3-3 6-4 10-2V4c-2-1-6-2-10 1Z',
    shield: 'm12 2 9 4v6c0 5-5 8-9 10-4-2-9-5-9-10V6Zm-4 10 3 3 5-6',
    menu: 'M4 6h16M4 12h16M4 18h16',
    close: 'm6 6 12 12M6 18 18 6',
    help: 'M9 9a3 3 0 0 1 6 0c0 2-3 2-3 5m0 3h.01M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0',
    check: 'm5 12 4 4L19 6',
    document: 'M14 2H5v20h14V7Zm0 0v5h5M8 12h8M8 16h6',
  };
}

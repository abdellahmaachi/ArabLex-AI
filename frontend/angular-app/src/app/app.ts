import { Component, ElementRef, HostListener, signal, viewChild } from '@angular/core';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import { Icon } from './shared/icon';
@Component({
  imports: [RouterLink, RouterLinkActive, RouterOutlet, Icon],
  selector: 'app-root',
  styleUrl: './app.scss',
  templateUrl: './app.html',
})
export class App {
  private readonly menuToggle = viewChild<ElementRef<HTMLButtonElement>>('menuToggle');
  readonly menuOpen = signal(false);
  readonly navigation = [
    { path: '/outils/recherche', label: 'Recherche lexicale', icon: 'search' },
    { path: '/outils/classification', label: 'Classification', icon: 'layers' },
    { path: '/outils/entites', label: 'Entités juridiques', icon: 'entities' },
    { path: '/outils/assistant', label: 'Assistant juridique', icon: 'spark' },
  ];
  @HostListener('document:keydown.escape')
  closeMenu(): void {
    if (this.menuOpen()) {
      this.menuOpen.set(false);
      this.menuToggle()?.nativeElement.focus();
    }
  }
}

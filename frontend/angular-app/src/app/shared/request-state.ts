import { HttpErrorResponse } from '@angular/common/http';
import { DestroyRef, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { finalize, Observable } from 'rxjs';

export function apiErrorMessage(error: unknown): string {
  if (!(error instanceof HttpErrorResponse)) return 'Une erreur inattendue est survenue.';
  if (error.status === 0) return 'Impossible de joindre l’API. Vérifiez que le serveur est démarré.';
  const detail: unknown = error.error?.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail.map((item) => `${(item.loc ?? []).join(' / ')} : ${item.msg ?? 'Valeur invalide'}`).join('\n');
  }
  return `La requête a échoué (HTTP ${error.status}). Réessayez.`;
}

export class RequestState<T> {
  private readonly destroyRef = inject(DestroyRef);
  readonly loading = signal(false);
  readonly error = signal('');
  readonly result = signal<T | null>(null);

  run(request: Observable<T>): void {
    if (this.loading()) return;
    this.loading.set(true);
    this.error.set('');
    this.result.set(null);
    request.pipe(
      takeUntilDestroyed(this.destroyRef),
      finalize(() => this.loading.set(false)),
    ).subscribe({
      next: (result) => this.result.set(result),
      error: (error: unknown) => this.error.set(apiErrorMessage(error)),
    });
  }
}

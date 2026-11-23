import { Component, input } from '@angular/core';

@Component({
  selector: 'app-request-status',
  template: `
    @if (loading()) { <p role="status">Traitement en cours…</p> }
    @if (error()) { <p role="alert" style="color: #a12121; white-space: pre-wrap">{{ error() }}</p> }
  `,
})
export class RequestStatus {
  readonly loading = input(false);
  readonly error = input('');
}

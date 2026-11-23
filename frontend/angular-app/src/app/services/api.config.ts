import { InjectionToken } from '@angular/core';

// Override in app.config.ts when deploying the API at a different address.
export const API_BASE_URL = new InjectionToken<string>('API_BASE_URL', {
  providedIn: 'root',
  factory: () => '/api',
});

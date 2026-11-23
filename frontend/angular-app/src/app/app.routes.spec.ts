import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { provideRouter, Router } from '@angular/router';
import { RouterTestingHarness } from '@angular/router/testing';
import { routes } from './app.routes';

describe('Service routes', () => {
  beforeEach(() => TestBed.configureTestingModule({
    providers: [provideRouter(routes), provideHttpClient(), provideHttpClientTesting()],
  }));
  afterEach(() => TestBed.inject(HttpTestingController).verify());

  it('redirects old bookmarks and renders a dedicated not-found page', async () => {
    const harness = await RouterTestingHarness.create('/rag');
    expect(TestBed.inject(Router).url).toBe('/outils/assistant');
    await harness.navigateByUrl('/outils/classification');
    expect(harness.routeNativeElement?.querySelector('app-classification')).toBeNull();
    expect(harness.routeNativeElement?.querySelector('#classification-text')).toBeTruthy();
    await harness.navigateByUrl('/page-inconnue');
    expect(harness.routeNativeElement?.textContent).toContain('Ce chemin ne mène à aucun outil.');
  });

  for (const [path, endpoint, field] of [
    ['/lexical', '/api/search/lexical', 'query'],
    ['/classification', '/api/classify', 'text'],
    ['/entities', '/api/ner/search', 'text'],
    ['/rag', '/api/rag/ask', 'question'],
  ]) {
    it(`${path} displays the form and submits to ${endpoint}`, async () => {
      const harness = await RouterTestingHarness.create(path);
      await harness.fixture.whenStable();
      const element = harness.routeNativeElement!;
      const http = TestBed.inject(HttpTestingController);
      http.expectNone(() => true);
      const input = element.querySelector('textarea')!;
      input.value = 'نص قانوني Legal text';
      input.dispatchEvent(new Event('input', { bubbles: true }));
      await harness.fixture.whenStable();
      expect(input.value).toBe('نص قانوني Legal text');
      expect(element.querySelector<HTMLButtonElement>('button[type="submit"]')!.disabled).toBe(false);
      const submitted = input.value;
      element.querySelector('form')!.dispatchEvent(new Event('submit', { cancelable: true }));
      const request = http.expectOne(endpoint);
      expect(request.request.method).toBe('POST');
      expect(request.request.body[field]).toBe(submitted);
      request.flush({ detail: 'Service temporairement indisponible' }, { status: 503, statusText: 'Unavailable' });
      await harness.fixture.whenStable();
      expect(element.querySelector('[role="alert"]')?.textContent).toContain('Service temporairement indisponible');
      if (path === '/rag') {
        expect(element.querySelector('.user-message')?.textContent).toContain(submitted);
      } else {
        expect(input.value).toBe(submitted);
      }
    });
  }
});

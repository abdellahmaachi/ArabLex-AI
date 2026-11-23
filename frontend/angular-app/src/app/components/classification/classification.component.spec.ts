import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { ClassificationComponent } from './classification.component';

describe('Classification form', () => {
  beforeEach(() => TestBed.configureTestingModule({
    imports: [ClassificationComponent],
    providers: [provideHttpClient(), provideHttpClientTesting()],
  }));
  afterEach(() => TestBed.inject(HttpTestingController).verify());

  it('rejects whitespace, shows loading and server errors, and allows retry', async () => {
    const fixture = TestBed.createComponent(ClassificationComponent);
    await fixture.whenStable();
    const element = fixture.nativeElement as HTMLElement;
    const text = element.querySelector('textarea')!;
    const form = element.querySelector('form')!;
    const button = element.querySelector('button')!;
    const http = TestBed.inject(HttpTestingController);
    text.value = '   ';
    text.dispatchEvent(new Event('input'));
    await fixture.whenStable();
    form.dispatchEvent(new Event('submit', { cancelable: true }));
    http.expectNone('/api/classify');
    expect(button.disabled).toBe(true);

    text.value = 'نص قانوني';
    text.dispatchEvent(new Event('input'));
    await fixture.whenStable();
    form.dispatchEvent(new Event('submit', { cancelable: true }));
    await fixture.whenStable();
    expect(button.disabled).toBe(true);
    expect(element.querySelector('[role="status"]')).toBeTruthy();
    form.dispatchEvent(new Event('submit', { cancelable: true }));
    http.expectOne('/api/classify').flush({ detail: 'Modèle indisponible' }, { status: 503, statusText: 'Unavailable' });
    await fixture.whenStable();
    expect(element.querySelector('[role="alert"]')?.textContent).toContain('Modèle indisponible');
    expect(button.disabled).toBe(false);

    form.dispatchEvent(new Event('submit', { cancelable: true }));
    http.expectOne('/api/classify').flush({ label: 'مدني', probabilities: { 'مدني': 0.9 } });
    await fixture.whenStable();
    expect(element.querySelector('[role="alert"]')).toBeNull();
    expect(element.querySelector('h3')?.textContent).toContain('مدني');
    expect(element.querySelector('table')?.textContent).toContain('90');
  });
});

import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { RAGComponent } from './rag.component';

describe('RAG conversation', () => {
  beforeEach(() => TestBed.configureTestingModule({
    imports: [RAGComponent], providers: [provideHttpClient(), provideHttpClientTesting()],
  }));
  afterEach(() => TestBed.inject(HttpTestingController).verify());

  it('retains turns, sources and the next draft; retries errors without duplicate messages', async () => {
    const fixture = TestBed.createComponent(RAGComponent);
    await fixture.whenStable();
    const element = fixture.nativeElement as HTMLElement;
    const http = TestBed.inject(HttpTestingController);
    const input = element.querySelector('textarea')!;
    const form = element.querySelector('form')!;
    const type = async (value: string) => {
      input.value = value;
      input.dispatchEvent(new Event('input', { bubbles: true }));
      await fixture.whenStable();
    };
    await type('ما هي شروط الإرث؟');
    form.dispatchEvent(new Event('submit', { cancelable: true }));
    const first = http.expectOne('/api/rag/ask');
    expect(first.request.body.question).toBe('ما هي شروط الإرث؟');
    expect(first.request.body.top_k).toBe(5);
    expect(fixture.componentInstance.loading()).toBe(true);
    await type('Next question');
    form.dispatchEvent(new Event('submit', { cancelable: true }));
    http.expectNone('/api/rag/ask');
    first.flush({ question: 'ما هي شروط الإرث؟', answer: 'جواب موثق [S1]', citation_status: 'valid', cited_references: ['S1'], generation: null,
      sources: [{ reference: 'S1', id: 'p1', score: .8, cited: true, passage: { document_id: 'document-1', text: 'نص المصدر' } }],
    });
    await fixture.whenStable();
    expect(input.value).toBe('Next question');
    expect(element.querySelector('.answer')?.textContent).toContain('جواب موثق');
    expect(element.querySelector('.source-card')?.textContent).toContain('نص المصدر');
    form.dispatchEvent(new Event('submit', { cancelable: true }));
    http.expectOne('/api/rag/ask').flush({ detail: 'Indisponible' }, { status: 503, statusText: 'Unavailable' });
    await fixture.whenStable();
    expect(fixture.componentInstance.turns()).toHaveLength(2);
    expect(element.querySelector('[role="alert"]')?.textContent).toContain('Indisponible');
    element.querySelector<HTMLButtonElement>('.chat-error button')!.click();
    const retry = http.expectOne('/api/rag/ask');
    expect(retry.request.body.question).toBe('Next question');
    retry.flush({ answer: 'Second answer', question: 'Next question', sources: [], cited_references: [], citation_status: 'not_applicable', generation: null });
    await fixture.whenStable();
    expect(fixture.componentInstance.turns()).toHaveLength(2);
    expect(element.querySelectorAll('.answer')).toHaveLength(2);
    element.querySelector<HTMLButtonElement>('.new-chat')!.click();
    await fixture.whenStable();
    expect(fixture.componentInstance.turns()).toHaveLength(0);
    expect(element.querySelector('.chat-welcome')).toBeTruthy();
  });
});

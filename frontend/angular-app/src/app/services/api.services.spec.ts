import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { API_BASE_URL } from './api.config';
import { LexicalSearchService } from './lexical-search.service';
import { ClassificationService } from './classification.service';
import { EntitySearchService } from './entity-search.service';
import { RAGService } from './rag.service';

describe('API services', () => {
  let http: HttpTestingController;
  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting(), { provide: API_BASE_URL, useValue: '/test/api/' }],
    });
    http = TestBed.inject(HttpTestingController);
  });
  afterEach(() => http.verify());
  it('posts the lexical-search contract', () => {
    const body = { query: 'نص قانوني' };
    let response: unknown;
    TestBed.inject(LexicalSearchService).search(body).subscribe(value => response = value);
    const request = http.expectOne('/test/api/search/lexical');
    expect(request.request.method).toBe('POST');
    expect(request.request.body).toEqual(body);
    const payload = { query: body.query, tokens: [], selected_class: null, class_scores: {}, results: [] };
    request.flush(payload);
    expect(response).toEqual(payload);
  });
  it('posts the classification contract', () => {
    const body = { text: 'نص قانوني' };
    let response: unknown;
    TestBed.inject(ClassificationService).classify(body).subscribe(value => response = value);
    const request = http.expectOne('/test/api/classify');
    expect(request.request.method).toBe('POST');
    expect(request.request.body).toEqual(body);
    const payload = { label: 'مدني', probabilities: { 'مدني': 1 } };
    request.flush(payload);
    expect(response).toEqual(payload);
  });
  it('posts the entity-search contract', () => {
    const body = { text: 'نص قانوني' };
    let response: unknown;
    TestBed.inject(EntitySearchService).search(body).subscribe(value => response = value);
    const request = http.expectOne('/test/api/ner/search');
    expect(request.request.method).toBe('POST');
    expect(request.request.body).toEqual(body);
    const payload = { text: body.text, results: [] };
    request.flush(payload);
    expect(response).toEqual(payload);
  });
  it('posts the rag contract', () => {
    const body = { question: 'نص قانوني' };
    let response: unknown;
    TestBed.inject(RAGService).ask(body).subscribe(value => response = value);
    const request = http.expectOne('/test/api/rag/ask');
    expect(request.request.method).toBe('POST');
    expect(request.request.body).toEqual(body);
    const payload = { question: body.question, answer: 'جواب', sources: [], cited_references: [], citation_status: 'not_applicable', generation: null };
    request.flush(payload);
    expect(response).toEqual(payload);
  });
});

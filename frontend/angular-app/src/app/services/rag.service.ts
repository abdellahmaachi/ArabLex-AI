import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { RAGRequest, RAGResponse } from '../models/api.models';
import { API_BASE_URL } from './api.config';

@Injectable({ providedIn: 'root' })
export class RAGService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = inject(API_BASE_URL).replace(/\/$/, '');
  ask(request: RAGRequest) {
    return this.http.post<RAGResponse>(`${this.baseUrl}/rag/ask`, request);
  }
}

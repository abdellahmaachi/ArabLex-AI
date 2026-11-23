import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { LexicalSearchRequest, LexicalSearchResponse } from '../models/api.models';
import { API_BASE_URL } from './api.config';

@Injectable({ providedIn: 'root' })
export class LexicalSearchService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = inject(API_BASE_URL).replace(/\/$/, '');
  search(request: LexicalSearchRequest) {
    return this.http.post<LexicalSearchResponse>(`${this.baseUrl}/search/lexical`, request);
  }
}

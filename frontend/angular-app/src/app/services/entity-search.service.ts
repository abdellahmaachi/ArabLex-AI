import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { EntitySearchRequest, EntitySearchResponse } from '../models/api.models';
import { API_BASE_URL } from './api.config';

@Injectable({ providedIn: 'root' })
export class EntitySearchService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = inject(API_BASE_URL).replace(/\/$/, '');
  search(request: EntitySearchRequest) {
    return this.http.post<EntitySearchResponse>(`${this.baseUrl}/ner/search`, request);
  }
}

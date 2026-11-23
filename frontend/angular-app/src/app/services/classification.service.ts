import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { ClassificationRequest, ClassificationResponse } from '../models/api.models';
import { API_BASE_URL } from './api.config';

@Injectable({ providedIn: 'root' })
export class ClassificationService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = inject(API_BASE_URL).replace(/\/$/, '');
  classify(request: ClassificationRequest) {
    return this.http.post<ClassificationResponse>(`${this.baseUrl}/classify`, request);
  }
}

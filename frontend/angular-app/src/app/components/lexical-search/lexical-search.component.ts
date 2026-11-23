import { ClassLabelPipe } from '../../shared/class-label.pipe';
import { ToolHeader } from '../../shared/tool-header';
import { Icon } from '../../shared/icon';
import { CommonModule } from '@angular/common';
import { Component, inject } from '@angular/core';
import { FormsModule, NgForm } from '@angular/forms';
import { LexicalSearchResponse } from '../../models/api.models';
import { LexicalSearchService } from '../../services/lexical-search.service';
import { RequestState } from '../../shared/request-state';
import { RequestStatus } from '../../shared/request-status';

@Component({
  selector: 'app-lexical-search',
  imports: [CommonModule, FormsModule, RequestStatus, ClassLabelPipe, ToolHeader, Icon],
  templateUrl: './lexical-search.component.html',
  styleUrl: '../../shared/feature.scss',
})
export class LexicalSearchComponent {
  private readonly api = inject(LexicalSearchService);
  readonly state = new RequestState<LexicalSearchResponse>();
  query = '';
  k = 10; minScore = 0; className = '';
  
  submit(form: NgForm): void {
    if (form.invalid || !this.query.trim() || this.state.loading()) return;
    this.state.run(this.api.search({ query: this.query, k: this.k, min_score: this.minScore, class_name: this.className.trim() || null }));
  }
}

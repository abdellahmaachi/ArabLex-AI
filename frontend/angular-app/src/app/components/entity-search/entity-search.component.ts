import { ClassLabelPipe } from '../../shared/class-label.pipe';
import { ToolHeader } from '../../shared/tool-header';
import { Icon } from '../../shared/icon';
import { CommonModule } from '@angular/common';
import { Component, inject } from '@angular/core';
import { FormsModule, NgForm } from '@angular/forms';
import { EntitySearchResponse } from '../../models/api.models';
import { EntitySearchService } from '../../services/entity-search.service';
import { RequestState } from '../../shared/request-state';
import { RequestStatus } from '../../shared/request-status';

@Component({
  selector: 'app-entity-search',
  imports: [CommonModule, FormsModule, RequestStatus, ClassLabelPipe, ToolHeader, Icon],
  templateUrl: './entity-search.component.html',
  styleUrl: '../../shared/feature.scss',
})
export class EntitySearchComponent {
  private readonly api = inject(EntitySearchService);
  readonly state = new RequestState<EntitySearchResponse>();
  text = '';
  k = 10; label = ''; className = '';
  
  submit(form: NgForm): void {
    if (form.invalid || !this.text.trim() || this.state.loading()) return;
    this.state.run(this.api.search({ text: this.text, k: this.k, label: this.label.trim() || null, class_name: this.className.trim() || null }));
  }
}

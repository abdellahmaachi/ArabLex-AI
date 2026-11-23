import { ClassLabelPipe } from '../../shared/class-label.pipe';
import { ToolHeader } from '../../shared/tool-header';
import { Icon } from '../../shared/icon';
import { CommonModule } from '@angular/common';
import { Component, inject } from '@angular/core';
import { FormsModule, NgForm } from '@angular/forms';
import { ClassificationResponse } from '../../models/api.models';
import { ClassificationService } from '../../services/classification.service';
import { RequestState } from '../../shared/request-state';
import { RequestStatus } from '../../shared/request-status';

@Component({
  selector: 'app-classification',
  imports: [CommonModule, FormsModule, RequestStatus, ClassLabelPipe, ToolHeader, Icon],
  templateUrl: './classification.component.html',
  styleUrl: '../../shared/feature.scss',
})
export class ClassificationComponent {
  private readonly api = inject(ClassificationService);
  readonly state = new RequestState<ClassificationResponse>();
  text = '';
  
  
  submit(form: NgForm): void {
    if (form.invalid || !this.text.trim() || this.state.loading()) return;
    this.state.run(this.api.classify({ text: this.text }));
  }
}

import { CommonModule } from '@angular/common';
import { afterEveryRender, Component, DestroyRef, ElementRef, inject, signal, viewChild } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormsModule, NgForm } from '@angular/forms';
import { finalize } from 'rxjs';
import { RAGRequest, RAGResponse } from '../../models/api.models';
import { RAGService } from '../../services/rag.service';
import { apiErrorMessage } from '../../shared/request-state';
import { Icon } from '../../shared/icon';

interface ChatTurn { id: number; request: RAGRequest; response?: RAGResponse; error?: string; }
@Component({
  selector: 'app-rag',
  imports: [CommonModule, FormsModule, Icon],
  templateUrl: './rag.component.html',
  styleUrl: './rag.component.scss',
})
export class RAGComponent {
  private readonly api = inject(RAGService);
  private readonly destroyRef = inject(DestroyRef);
  private readonly conversation = viewChild<ElementRef<HTMLElement>>('conversation');
  private readonly composer = viewChild<ElementRef<HTMLTextAreaElement>>('composer');
  private nextId = 0;
  private scrollRequested = false;
  readonly turns = signal<ChatTurn[]>([]);
  readonly loading = signal(false);
  question = '';
  topK = 5;
  documentId = '';
  className = '';
  scoreThreshold: number | null = null;
  readonly suggestions = ['ما هي شروط الإرث؟', 'كيف يتم إثبات الملكية العقارية؟', 'ما هي حقوق الزوجين؟'];
  readonly citationLabels = { valid: 'Citations vérifiées', missing: 'Citations manquantes', not_applicable: 'Citations non applicables' };

  constructor() {
    afterEveryRender(() => {
      if (this.scrollRequested) {
        const element = this.conversation()?.nativeElement;
        if (element) element.scrollTop = element.scrollHeight;
        this.scrollRequested = false;
      }
    });
  }

  useSuggestion(text: string): void {
    this.question = text;
    this.composer()?.nativeElement.focus();
  }

  submit(form: NgForm): void {
    if (form.invalid || !this.question.trim() || this.loading()) return;
    const turn: ChatTurn = { id: this.nextId++, request: {
      question: this.question, top_k: this.topK,
      document_id: this.documentId.trim() || null,
      class_name: this.className.trim() || null, score_threshold: this.scoreThreshold,
    } };
    this.turns.update(turns => [...turns, turn]);
    this.question = '';
    this.send(turn);
  }

  onKeydown(event: KeyboardEvent, form: NgForm): void {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      this.submit(form);
    }
  }

  retry(turn: ChatTurn): void {
    if (this.loading()) return;
    this.updateTurn(turn.id, { error: undefined });
    this.send(turn);
  }

  newConversation(): void {
    if (this.loading()) return;
    this.turns.set([]);
    this.question = '';
    this.composer()?.nativeElement.focus();
  }

  private send(turn: ChatTurn): void {
    this.loading.set(true);
    this.scrollRequested = true;
    this.api.ask(turn.request).pipe(
      takeUntilDestroyed(this.destroyRef),
      finalize(() => this.loading.set(false)),
    ).subscribe({
      next: response => { this.updateTurn(turn.id, { response, error: undefined }); this.scrollRequested = true; },
      error: error => { this.updateTurn(turn.id, { error: apiErrorMessage(error) }); this.scrollRequested = true; },
    });
  }

  private updateTurn(id: number, patch: Partial<ChatTurn>): void {
    this.turns.update(turns => turns.map(turn => turn.id === id ? { ...turn, ...patch } : turn));
  }
}

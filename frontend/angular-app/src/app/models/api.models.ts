export interface LegalDocument {
  id: string;
  class: string;
  filename: string;
  path: string;
  text: string;
}

export interface LexicalSearchRequest {
  query: string;
  k?: number;
  min_score?: number;
  class_name?: string | null;
}
export interface LexicalSearchResponse {
  query: string;
  tokens: string[];
  selected_class: string | null;
  class_scores: Record<string, number>;
  results: (LegalDocument & { score: number })[];
}
export interface ClassificationRequest { text: string; }
export interface ClassificationResponse {
  label: string | number;
  probabilities: Record<string, number>;
}
export interface Entity {
  text: string;
  label: string;
  start: number;
  end: number;
}
export interface EntitySearchRequest {
  text: string;
  label?: string | null;
  class_name?: string | null;
  k?: number;
}
export interface EntitySearchResponse {
  text: string;
  results: (LegalDocument & { entities: Entity[] })[];
}
export interface RAGRequest {
  question: string;
  top_k?: number;
  document_id?: string | null;
  class_name?: string | null;
  score_threshold?: number | null;
}
export interface Passage {
  id: string;
  document_id: string;
  chunk_index: number;
  text: string;
  text_field: string;
  start: number;
  end: number;
  word_count: number;
  metadata: Record<string, unknown>;
}
export interface RAGSource {
  reference: string;
  id: string;
  score: number;
  passage: Passage;
  cited: boolean;
}
export interface RAGResponse {
  question: string;
  answer: string;
  sources: RAGSource[];
  cited_references: string[];
  citation_status: 'valid' | 'missing' | 'not_applicable';
  generation: {
    model: string;
    finish_reason: string;
    prompt_tokens: number | null;
    output_tokens: number | null;
    total_tokens: number | null;
  } | null;
}

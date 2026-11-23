import { Pipe, PipeTransform } from '@angular/core';

// Presentation labels only: API payloads retain their original identifiers.
export const ARABIC_CLASS_LABELS: Record<string, string> = {
  family_inheritance: 'الأسرة والميراث',
  real_estate: 'العقار',
};

@Pipe({ name: 'classLabel' })
export class ClassLabelPipe implements PipeTransform {
  transform(value: string | number | null): string {
    if (value === null) return 'غير محدد';
    return ARABIC_CLASS_LABELS[String(value)] ?? String(value);
  }
}

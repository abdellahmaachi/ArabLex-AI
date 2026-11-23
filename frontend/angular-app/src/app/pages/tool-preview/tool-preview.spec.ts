import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { RouterTestingHarness } from '@angular/router/testing';
import { ToolPreview } from './tool-preview';

describe('Editable tool previews', () => {
  beforeEach(() => TestBed.configureTestingModule({ providers: [provideRouter(
    ['lexical', 'entities', 'classification', 'rag'].map(path => ({
      path, component: ToolPreview,
      data: { tool: { title: path, icon: 'search', description: '', example: '' } },
    })),
  )] }));

  for (const [path, tag] of [
    ['/lexical', 'INPUT'], ['/entities', 'INPUT'],
    ['/classification', 'TEXTAREA'], ['/rag', 'TEXTAREA'],
  ]) {
    it(`${path} accepts focus and preserves Arabic and English while typing`, async () => {
      // No HttpClient provider: these routes must remain usable without the API.
      const harness = await RouterTestingHarness.create();
      const component = await harness.navigateByUrl(path, ToolPreview);
      await harness.fixture.whenStable();
      const field = harness.routeNativeElement!.querySelector<HTMLInputElement | HTMLTextAreaElement>('#tool-text')!;
      expect(field.tagName).toBe(tag);
      expect(field.disabled).toBe(false);
      expect(field.readOnly).toBe(false);
      expect(field.getAttribute('dir')).toBe('auto');
      field.focus();
      expect(document.activeElement).toBe(field);

      for (const sample of ['المسؤولية المدنية', 'Legal research', 'العقد contract 123']) {
        field.value = '';
        field.dispatchEvent(new Event('input', { bubbles: true }));
        let typed = '';
        for (const character of sample) {
          typed += character;
          field.value = typed;
          field.dispatchEvent(new Event('input', { bubbles: true }));
          await harness.fixture.whenStable();
          harness.detectChanges();
          expect(component.text).toBe(typed);
          expect(field.value).toBe(typed);
          expect(document.activeElement).toBe(field);
          expect(harness.routeNativeElement!.querySelector('#tool-text')).toBe(field);
        }
      }

      field.blur();
      await harness.fixture.whenStable();
      field.focus();
      expect(field.value).toBe('العقد contract 123');
    });
  }
});

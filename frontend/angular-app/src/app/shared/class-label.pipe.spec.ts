import { ClassLabelPipe } from './class-label.pipe';

describe('Arabic classification labels', () => {
  const pipe = new ClassLabelPipe();
  it('translates the two exported model classes', () => {
    expect(pipe.transform('family_inheritance')).toBe('الأسرة والميراث');
    expect(pipe.transform('real_estate')).toBe('العقار');
  });
  it('preserves unrecognized and already Arabic labels without inventing a category', () => {
    expect(pipe.transform('مدني')).toBe('مدني');
    expect(pipe.transform('new_class')).toBe('new_class');
  });
});

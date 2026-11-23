import { isAbsolute, relative } from 'node:path';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  plugins: [
    {
      name: 'relative-angular-test-paths',
      // Angular supplies absolute filenames as glob patterns. Parentheses in
      // an ancestor directory (e.g. Master(MLAIM)) prevent Vitest matching them.
      // Run before Vitest consumes the resolved project configuration, keeping
      // Angular's discovered file list (including CLI include/exclude filters).
      configResolved: {
        order: 'pre',
        handler(config) {
          if (config.test?.include) {
            config.test.include = config.test.include.map((file) =>
              isAbsolute(file)
                ? relative(config.test.dir ?? config.root, file).replaceAll('\\', '/')
                : file,
            );
          }
        },
      },
    },
  ],
});

// Builds the UI into dist/: esbuild bundles the React app, Tailwind compiles only the classes
// the source actually uses. No Vite or Tailwind CLI needed. `node build.mjs --watch` rebuilds
// on change; run the API with `uv run python -m app serve --reload` and reload the browser.
import { build, context } from 'esbuild';
import { compile } from 'tailwindcss';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';

const here = path.dirname(new URL(import.meta.url).pathname);
const src = path.join(here, 'src');
const dist = path.join(here, 'dist');
const require = createRequire(import.meta.url);
const watch = process.argv.includes('--watch');

function sourceFiles(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((e) =>
    e.isDirectory() ? sourceFiles(path.join(dir, e.name)) : [path.join(dir, e.name)]);
}

async function buildCss() {
  const input = fs.readFileSync(path.join(src, 'styles.css'), 'utf8');
  const twRoot = path.dirname(require.resolve('tailwindcss/package.json'));
  const compiler = await compile(input, {
    base: src,
    loadStylesheet: async (id, base) => {
      const file = id === 'tailwindcss' ? path.join(twRoot, 'index.css') : path.resolve(base, id);
      return { path: file, base: path.dirname(file), content: fs.readFileSync(file, 'utf8') };
    },
  });
  const candidates = new Set();
  for (const f of sourceFiles(src).filter((f) => /\.(tsx?|html)$/.test(f))) {
    // Split on whitespace and quote-like delimiters only: arbitrary values such as
    // w-[min(420px,100%)] contain parentheses and commas that must survive.
    for (const tok of fs.readFileSync(f, 'utf8').split(/[\s"'`{}<>;]+/)) if (tok) candidates.add(tok);
  }
  fs.writeFileSync(path.join(dist, 'app.css'), compiler.build([...candidates]));
}

function copyHtml() {
  fs.copyFileSync(path.join(here, 'index.html'), path.join(dist, 'index.html'));
}

const options = {
  entryPoints: [path.join(src, 'main.tsx')],
  bundle: true,
  outfile: path.join(dist, 'app.js'),
  format: 'esm',
  target: ['safari17', 'chrome120'],
  jsx: 'automatic',
  minify: !watch,
  sourcemap: watch ? 'inline' : false,
  define: { 'process.env.NODE_ENV': watch ? '"development"' : '"production"' },
  logLevel: 'info',
};

fs.mkdirSync(dist, { recursive: true });
copyHtml();
await buildCss();
if (watch) {
  const ctx = await context({
    ...options,
    plugins: [{ name: 'css', setup(b) { b.onEnd(() => buildCss().then(copyHtml).catch(console.error)); } }],
  });
  await ctx.watch();
  console.log('watching src/ for changes');
} else {
  await build(options);
}

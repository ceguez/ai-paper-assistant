import DOMPurify from 'dompurify';
import { marked } from 'marked';
import { useMemo } from 'react';
import type { Source } from '../types';

const CITE = /\[(n:[0-9a-f]{6,24}(?:\s*[,;]\s*n:[0-9a-f]{6,24})*)\]/g;

/** Number citations in order of first appearance and turn them into highlighter chips. */
export function citationOrder(text: string): string[] {
  const order: string[] = [];
  for (const m of text.matchAll(CITE)) {
    for (const id of m[1].split(/[,;]/).map((s) => s.trim().slice(2))) {
      if (!order.includes(id)) order.push(id);
    }
  }
  return order;
}

interface Props {
  text: string;
  sources: Source[];
  streaming: boolean;
  activeNode: string | null;
  onCite: (nodeId: string) => void;
}

export function Answer({ text, sources, streaming, activeNode, onCite }: Props) {
  const html = useMemo(() => {
    const order = citationOrder(text);
    const known = new Set(sources.map((s) => s.node_id));
    const withChips = text.replace(CITE, (_, ids: string) =>
      ids.split(/[,;]/).map((raw) => {
        const id = raw.trim().slice(2);
        const n = order.indexOf(id) + 1;
        const src = sources.find((s) => s.node_id === id);
        const label = src ? `${src.paper_title}: ${src.breadcrumb.split(' > ').slice(1).join(' > ')}` : 'Source';
        return known.has(id)
          ? `<button class="cite" data-node="${id}" aria-pressed="${id === activeNode}" title="${label.replace(/"/g, '&quot;')}">${n}</button>`
          : '';  // a citation to something the model never read is dropped, not shown
      }).join(''));
    return DOMPurify.sanitize(marked.parse(withChips, { async: false }) as string);
  }, [text, sources, activeNode]);

  return (
    <div
      className={`prose-answer ${streaming ? 'caret' : ''}`}
      onClick={(e) => {
        const el = (e.target as HTMLElement).closest('button.cite') as HTMLElement | null;
        if (el?.dataset.node) onCite(el.dataset.node);
      }}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

import type { Message, RunEvent, Source } from '../types';
import { citationOrder } from './Answer';

export type Tab = 'sources' | 'trail' | 'reader';

export interface ReaderTarget {
  paperId: string;
  title: string;
  page: number | null;
}

interface Props {
  tab: Tab;
  onTab: (t: Tab) => void;
  message: Message | null;
  activeNode: string | null;
  onOpenSource: (s: Source) => void;
  reader: ReaderTarget | null;
  onClose?: () => void;
}

const TABS: { id: Tab; label: string }[] = [
  { id: 'sources', label: 'Sources' },
  { id: 'trail', label: 'Research trail' },
  { id: 'reader', label: 'PDF' },
];

export function SidePane(props: Props) {
  return (
    <section className="flex h-full min-h-0 flex-col border-l border-rule bg-sheet" aria-label="Details">
      <div role="tablist" className="flex items-end gap-4 border-b border-rule px-4 pt-3">
        {TABS.map((t) => (
          <button
            key={t.id}
            role="tab"
            aria-selected={props.tab === t.id}
            className={`-mb-px border-b-2 pb-2 text-[13px] ${props.tab === t.id ? 'border-ink font-semibold' : 'border-transparent text-ink-soft hover:text-ink'}`}
            onClick={() => props.onTab(t.id)}
          >
            {t.label}
            {t.id === 'sources' && props.message?.sources.length ? (
              <span className="ml-1.5 text-ink-soft">{props.message.sources.length}</span>
            ) : null}
          </button>
        ))}
        {props.onClose && (
          <button className="ml-auto pb-2 text-[13px] text-ink-soft hover:text-ink" onClick={props.onClose}>Close</button>
        )}
      </div>
      <div className="scroll-thin min-h-0 flex-1 overflow-y-auto">
        {props.tab === 'sources' && <Sources {...props} />}
        {props.tab === 'trail' && <Trail events={props.message?.trail ?? []} />}
        {props.tab === 'reader' && <Reader target={props.reader} />}
      </div>
    </section>
  );
}

function Sources({ message, activeNode, onOpenSource }: Props) {
  if (!message || !message.sources.length) {
    return (
      <p className="p-4 text-[13px] text-ink-soft">
        Sections the assistant reads appear here, with the page they come from. Click a highlighted number in an
        answer to jump to its source.
      </p>
    );
  }
  const order = citationOrder(message.content);
  const sorted = [...message.sources].sort((a, b) => {
    const ia = order.indexOf(a.node_id), ib = order.indexOf(b.node_id);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
  });
  return (
    <ol className="divide-y divide-rule">
      {sorted.map((s) => {
        const n = order.indexOf(s.node_id) + 1;
        const path = s.breadcrumb.split(' > ').slice(1).join(' > ') || 'Front matter';
        return (
          <li key={s.node_id} id={`src-${s.node_id}`}
              className={`px-4 py-3 ${s.node_id === activeNode ? 'bg-mark-wash' : ''}`}>
            <div className="flex items-start gap-2">
              {n > 0 ? <span className="cite !ml-0 !align-[0.1em]">{n}</span>
                     : <span className="mt-0.5 text-xs text-ink-soft" title="Read but not cited">read</span>}
              <div className="min-w-0 flex-1">
                <div className="text-[13px] font-medium leading-snug">{s.paper_title}</div>
                <div className="text-xs text-ink-soft">{path}</div>
              </div>
            </div>
            <p className={`mt-2 border-l-[3px] pl-3 font-[family-name:var(--font-reading)] text-[14px] leading-relaxed ${n > 0 ? 'border-mark' : 'border-rule'}`}>
              {s.preview}{s.preview.length >= 280 ? '…' : ''}
            </p>
            <button className="mt-2 text-xs font-medium text-action hover:underline" onClick={() => onOpenSource(s)}>
              {s.page ? `Open page ${s.page}` : 'Open PDF'}
            </button>
          </li>
        );
      })}
    </ol>
  );
}

function describe(ev: RunEvent): { head: string; body?: string; tone?: 'error' | 'quiet' } | null {
  const d = ev.data;
  switch (ev.type) {
    case 'route':
      return { head: `Treated as ${({ chat: 'conversation', search: 'a search', deep_read: 'a close reading', compare: 'a comparison' } as Record<string, string>)[d.route] ?? d.route}`,
               body: d.reasoning };
    case 'tool_call':
      if (d.tool === 'search_papers') return { head: `Searched for “${d.query}”`, body: d.reasoning !== 'seed retrieval' ? d.reasoning : undefined };
      if (d.tool === 'read_section') return { head: 'Opened a section', body: d.reasoning };
      if (d.tool === 'get_element') return { head: 'Looked at a table or figure', body: d.reasoning };
      if (d.tool === 'save_note') return { head: 'Took a note', body: d.claim || d.reasoning };
      return null;
    case 'source':
      return { head: `Read ${d.breadcrumb.split(' > ').slice(-1)[0]}`, body: d.paper_title, tone: 'quiet' };
    case 'verify':
      if (d.skipped) return { head: 'Skipped the fact check', body: d.reason, tone: 'quiet' };
      return d.unsupported?.length
        ? { head: `Checked ${d.checked} claims, ${d.unsupported.length} not supported`, body: d.unsupported.join('; '), tone: 'error' }
        : { head: `Checked ${d.checked} ${d.checked === 1 ? 'claim' : 'claims'} against the sources` };
    case 'error':
      return { head: `Problem during ${d.node}; switched to plain search`, body: d.message, tone: 'error' };
    case 'step':
      return d.phase === 'note' ? { head: d.note, tone: 'quiet' } : null;
    case 'done':
      return { head: d.status === 'done' ? 'Finished' : d.status === 'cancelled' ? 'Stopped' : 'Failed',
               body: d.errors?.length && d.status !== 'done' ? d.errors.join('; ') : undefined, tone: 'quiet' };
    default:
      return null;
  }
}

function Trail({ events }: { events: RunEvent[] }) {
  const rows = events.map((ev) => ({ ev, row: describe(ev) })).filter((x) => x.row);
  if (!rows.length) {
    return <p className="p-4 text-[13px] text-ink-soft">Each step the assistant takes to answer is recorded here.</p>;
  }
  const t0 = events[0]?.ts ?? 0;
  return (
    <ol className="px-4 py-3">
      {rows.map(({ ev, row }) => (
        <li key={ev.seq} className="grid grid-cols-[3.2rem_1fr] gap-2 py-1.5">
          <span className="pt-px text-right text-xs tabular-nums text-ink-soft">{(ev.ts - t0).toFixed(1)}s</span>
          <div className={`border-l pl-3 ${row!.tone === 'error' ? 'border-danger' : 'border-rule'}`}>
            <div className={`text-[13px] ${row!.tone === 'quiet' ? 'text-ink-soft' : row!.tone === 'error' ? 'text-danger' : ''}`}>
              {row!.head}
            </div>
            {row!.body && <div className="text-xs text-ink-soft">{row!.body}</div>}
          </div>
        </li>
      ))}
    </ol>
  );
}

function Reader({ target }: { target: ReaderTarget | null }) {
  if (!target) {
    return <p className="p-4 text-[13px] text-ink-soft">Open a source or a paper from the library to read it here.</p>;
  }
  const src = `/api/papers/${target.paperId}/file#page=${target.page ?? 1}`;
  return (
    <div className="flex h-full flex-col">
      <div className="flex items-baseline justify-between gap-3 px-4 py-2 text-xs text-ink-soft">
        <span className="truncate">{target.title}{target.page ? `, page ${target.page}` : ''}</span>
        <a className="shrink-0 text-action hover:underline" href={src} target="_blank" rel="noreferrer">Open in new tab</a>
      </div>
      {/* key forces a reload so the built-in PDF viewer honours the new #page */}
      <iframe key={src} title={target.title} src={src} className="min-h-0 w-full flex-1 border-0 bg-paper" />
    </div>
  );
}

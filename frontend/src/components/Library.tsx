import { useRef, useState } from 'react';
import type { Paper, Session } from '../types';

interface Props {
  papers: Paper[];
  scope: string[];
  onToggleScope: (id: string) => void;
  onClearScope: () => void;
  onUpload: (files: File[]) => void;
  onDelete: (p: Paper) => void;
  onRetry: (p: Paper) => void;
  onOpenPaper: (p: Paper) => void;
  sessions: Session[];
  currentSession: string | null;
  onSelectSession: (id: string) => void;
  onNewChat: () => void;
}

function statusLine(p: Paper): string {
  if (p.status === 'ready') {
    const bits = [p.authors?.split(',')[0], p.year].filter(Boolean);
    return bits.length ? bits.join(', ') : `${p.page_count ?? '?'} pages`;
  }
  if (p.status === 'failed') return p.error || 'Could not read this PDF';
  if (p.status === 'queued') return 'Waiting to be read';
  return p.progress ? `Reading: ${p.progress}` : 'Reading';
}

export function Library(props: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const ready = props.papers.filter((p) => p.status === 'ready').length;

  const takeFiles = (list: FileList | null) => {
    const files = Array.from(list ?? []).filter((f) => f.name.toLowerCase().endsWith('.pdf'));
    if (files.length) props.onUpload(files);
  };

  return (
    <aside
      className={`flex h-full min-h-0 flex-col border-r border-rule bg-sheet ${dragging ? 'outline-2 outline-dashed outline-action -outline-offset-8' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => { e.preventDefault(); setDragging(false); takeFiles(e.dataTransfer.files); }}
    >
      <div className="flex items-baseline justify-between px-4 pt-4 pb-2">
        <h2 className="text-[15px] font-semibold">Library</h2>
        <span className="text-xs text-ink-soft">{ready} {ready === 1 ? 'paper' : 'papers'}</span>
      </div>
      <div className="px-4 pb-3">
        <button
          className="w-full rounded-md border border-dashed border-rule px-3 py-2.5 text-left text-[13px] text-ink-soft hover:border-action hover:text-ink"
          onClick={() => input.current?.click()}
        >
          <span className="font-medium text-ink">Add PDFs</span> or drop them here
        </button>
        <input ref={input} type="file" accept="application/pdf,.pdf" multiple hidden
               onChange={(e) => { takeFiles(e.target.files); e.target.value = ''; }} />
      </div>

      {props.scope.length > 0 && (
        <div className="mx-4 mb-2 flex items-center justify-between rounded-md bg-mark-wash px-2.5 py-1.5 text-xs">
          <span>Answers use {props.scope.length} selected {props.scope.length === 1 ? 'paper' : 'papers'}</span>
          <button className="font-medium underline-offset-2 hover:underline" onClick={props.onClearScope}>Use all</button>
        </div>
      )}

      <ul className="scroll-thin min-h-0 flex-1 overflow-y-auto px-2" aria-label="Papers">
        {props.papers.length === 0 && (
          <li className="px-2 py-6 text-[13px] text-ink-soft">
            Add a few papers to start. Each one is split into its sections so answers can quote the exact part.
          </li>
        )}
        {props.papers.map((p) => {
          const inScope = props.scope.includes(p.paper_id);
          return (
            <li key={p.paper_id} className="group flex gap-2 rounded-md px-2 py-2 hover:bg-paper">
              <input
                type="checkbox"
                className="mt-1 accent-[var(--action)]"
                aria-label={`Limit answers to ${p.title}`}
                checked={inScope}
                disabled={p.status !== 'ready'}
                onChange={() => props.onToggleScope(p.paper_id)}
              />
              <div className="min-w-0 flex-1">
                <button
                  className="line-clamp-2 block w-full text-left text-[13px] leading-snug font-medium disabled:text-ink-soft"
                  title={p.title ?? p.filename ?? ''}
                  disabled={p.status !== 'ready'}
                  onClick={() => props.onOpenPaper(p)}
                >
                  {p.title || p.filename}
                </button>
                <div className={`truncate text-xs ${p.status === 'failed' ? 'text-danger' : 'text-ink-soft'}`}
                     title={statusLine(p)}>
                  {(p.status === 'processing' || p.status === 'queued') && (
                    <span className="mr-1.5 inline-block size-1.5 animate-pulse rounded-full bg-action align-middle" />
                  )}
                  {statusLine(p)}
                </div>
              </div>
              <div className={`shrink-0 items-start gap-1 ${p.status === 'failed' ? 'flex' : 'hidden group-hover:flex group-focus-within:flex'}`}>
                {p.status === 'failed' && (
                  <button className="rounded px-1.5 text-xs text-action hover:underline" onClick={() => props.onRetry(p)}>
                    Retry
                  </button>
                )}
                <button className="rounded px-1.5 text-xs text-ink-soft hover:text-danger" onClick={() => props.onDelete(p)}
                        aria-label={`Remove ${p.title}`}>
                  Remove
                </button>
              </div>
            </li>
          );
        })}
      </ul>

      <div className="border-t border-rule">
        <div className="flex items-baseline justify-between px-4 pt-3 pb-1">
          <h2 className="text-[15px] font-semibold">Chats</h2>
          <button className="text-xs font-medium text-action hover:underline" onClick={props.onNewChat}>New chat</button>
        </div>
        <ul className="scroll-thin max-h-48 overflow-y-auto px-2 pb-3" aria-label="Chats">
          {props.sessions.filter((s) => s.title).map((s) => (
            <li key={s.session_id}>
              <button
                className={`block w-full truncate rounded-md px-2 py-1.5 text-left text-[13px] ${s.session_id === props.currentSession ? 'bg-paper font-medium' : 'text-ink-soft hover:bg-paper hover:text-ink'}`}
                onClick={() => props.onSelectSession(s.session_id)}
                aria-current={s.session_id === props.currentSession}
              >
                {s.title}
              </button>
            </li>
          ))}
        </ul>
      </div>
    </aside>
  );
}

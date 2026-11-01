import { useEffect, useRef, useState } from 'react';
import type { Message, Paper } from '../types';
import { Answer } from './Answer';

interface Props {
  messages: Message[];
  focused: string | null;
  activeNode: string | null;
  running: boolean;
  scopeTitles: string[];
  readyPapers: Paper[];
  modelReady: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
  onFocus: (key: string) => void;
  onCite: (msgKey: string, nodeId: string) => void;
}

const STARTERS = [
  'What problem does each paper set out to solve?',
  'Summarise the method of the most recent paper',
  'Which papers report results on the same benchmark?',
];

export function Chat(props: Props) {
  const [draft, setDraft] = useState('');
  const end = useRef<HTMLDivElement>(null);
  const box = useRef<HTMLTextAreaElement>(null);
  const last = props.messages.at(-1);

  useEffect(() => {
    end.current?.scrollIntoView({ block: 'end' });
  }, [props.messages.length, last?.content.length]);

  const send = () => {
    const text = draft.trim();
    if (!text || props.running) return;
    props.onSend(text);
    setDraft('');
  };

  const empty = props.messages.length === 0;
  return (
    <main className="flex h-full min-h-0 flex-col">
      <div className="scroll-thin min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-[760px] px-6 py-8">
          {empty && (
            <div className="pt-[12vh]">
              <h1 className="font-[family-name:var(--font-reading)] text-[30px] leading-tight">
                {props.readyPapers.length ? 'Ask your papers.' : 'Start by adding a paper.'}
              </h1>
              <p className="mt-2 max-w-[56ch] text-[15px] text-ink-soft">
                {props.readyPapers.length
                  ? 'Answers quote the sections they rely on, marked like a highlighter, with the page to check.'
                  : 'Drop PDFs into the library on the left. Everything stays on this Mac: the model, the index and your notes.'}
              </p>
              {props.readyPapers.length > 0 && (
                <div className="mt-6 flex flex-col items-start gap-2">
                  {STARTERS.map((s) => (
                    <button key={s} className="rounded-md border border-rule bg-sheet px-3 py-2 text-left text-[13px] hover:border-action"
                            onClick={() => props.onSend(s)} disabled={props.running || !props.modelReady}>
                      {s}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
          {props.messages.map((m) =>
            m.role === 'user' ? (
              <div key={m.key} className="mt-8 mb-3 flex justify-end first:mt-0">
                <div className="max-w-[80%] rounded-lg bg-ink px-3.5 py-2 text-[14px] whitespace-pre-wrap text-paper">{m.content}</div>
              </div>
            ) : (
              <article
                key={m.key}
                className={`-mx-3 rounded-lg px-3 py-2 ${props.focused === m.key ? 'bg-sheet' : ''}`}
                onClick={() => props.onFocus(m.key)}
                aria-busy={m.status === 'streaming'}
              >
                {m.content ? (
                  <Answer text={m.content} sources={m.sources} streaming={m.status === 'streaming'}
                          activeNode={props.focused === m.key ? props.activeNode : null}
                          onCite={(id) => props.onCite(m.key, id)} />
                ) : (
                  <p className="text-[13px] text-ink-soft">{m.status === 'streaming' ? progressLine(m) : 'No answer.'}</p>
                )}
                {m.status !== 'streaming' && m.status !== 'done' && (
                  <p className={`mt-1 text-xs ${m.status === 'failed' ? 'text-danger' : 'text-ink-soft'}`}>
                    {m.status === 'cancelled' ? 'Stopped before finishing.' : 'The model could not finish this answer. Check that Ollama is running, then ask again.'}
                  </p>
                )}
              </article>
            ),
          )}
          <div ref={end} />
        </div>
      </div>

      <div className="border-t border-rule bg-paper">
        <div className="mx-auto w-full max-w-[760px] px-6 py-3">
          <div className="rounded-lg border border-rule bg-sheet focus-within:border-action">
            <textarea
              ref={box}
              rows={2}
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); send(); }
              }}
              placeholder={props.modelReady ? 'Ask about your papers' : 'Waiting for the local model…'}
              aria-label="Question"
              className="block w-full resize-none bg-transparent px-3 pt-2.5 text-[15px] outline-none placeholder:text-ink-soft"
            />
            <div className="flex items-center justify-between gap-3 px-3 pb-2">
              <span className="truncate text-xs text-ink-soft">
                {props.scopeTitles.length
                  ? `Only in: ${props.scopeTitles.join('; ')}`
                  : `Searching all ${props.readyPapers.length} ${props.readyPapers.length === 1 ? 'paper' : 'papers'}`}
              </span>
              {props.running ? (
                <button className="shrink-0 rounded-md border border-rule px-3 py-1 text-[13px] hover:border-danger hover:text-danger" onClick={props.onStop}>
                  Stop
                </button>
              ) : (
                <button className="shrink-0 rounded-md bg-action px-3 py-1 text-[13px] font-medium text-action-ink disabled:opacity-40"
                        onClick={send} disabled={!draft.trim() || !props.modelReady}>
                  Ask
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}

function progressLine(m: Message): string {
  const lastStep = [...m.trail].reverse().find((e) => e.type === 'tool_call' || e.type === 'route' || e.type === 'step');
  if (!lastStep) return 'Thinking…';
  if (lastStep.type === 'tool_call' && lastStep.data.tool === 'search_papers') return `Searching for “${lastStep.data.query}”…`;
  if (lastStep.type === 'tool_call') return 'Reading…';
  if (lastStep.type === 'step' && lastStep.data.node === 'answer') return 'Writing…';
  if (lastStep.type === 'step' && lastStep.data.node === 'route') return 'Reading your question…';
  return `Reading ${m.sources.length ? `${m.sources.length} sections` : 'the library'}…`;
}

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api, applyEvent, EVENT_TYPES } from './api';
import { Chat } from './components/Chat';
import { Library } from './components/Library';
import { SidePane, type ReaderTarget, type Tab } from './components/SidePane';
import type { Health, Message, Paper, RunEvent, Session, Source } from './types';

const remember = {
  get: (k: string) => { try { return localStorage.getItem(k); } catch { return null; } },
  set: (k: string, v: string) => { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
};

let keySeq = 0;
const newKey = () => `m${Date.now()}-${keySeq++}`;

export function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [papers, setPapers] = useState<Paper[]>([]);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [session, setSession] = useState<string | null>(remember.get('pa.session'));
  const [messages, setMessages] = useState<Message[]>([]);
  const [scope, setScope] = useState<string[]>([]);
  const [focused, setFocused] = useState<string | null>(null);
  const [activeNode, setActiveNode] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>('sources');
  const [reader, setReader] = useState<ReaderTarget | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [showLibrary, setShowLibrary] = useState(false);
  const [showPane, setShowPane] = useState(false);
  const stream = useRef<EventSource | null>(null);
  const createdHere = useRef<string | null>(null); // a session send() just made: nothing to load

  // ---------- polling ----------
  useEffect(() => {
    let alive = true;
    const tick = () => api.health().then((h) => { if (alive) { setHealth(h); setHealthError(false); } })
      .catch(() => alive && setHealthError(true));
    tick();
    const id = setInterval(tick, 5000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  const busyIngest = papers.some((p) => p.status === 'queued' || p.status === 'processing');
  const loadPapers = useCallback(() => api.papers().then(setPapers).catch(() => {}), []);
  useEffect(() => {
    loadPapers();
    const id = setInterval(loadPapers, busyIngest ? 1500 : 8000);
    return () => clearInterval(id);
  }, [busyIngest, loadPapers]);

  const loadSessions = useCallback(() => api.sessions().then(setSessions).catch(() => {}), []);
  useEffect(() => { loadSessions(); }, [loadSessions]);

  // ---------- session history (sources and trail are rebuilt from each run's saved events) ----------
  useEffect(() => {
    if (session && createdHere.current === session) {
      createdHere.current = null;
      remember.set('pa.session', session);
      return;
    }
    stream.current?.close();
    setRunId(null);
    setActiveNode(null);
    if (!session) { setMessages([]); return; }
    remember.set('pa.session', session);
    let alive = true;
    api.messages(session).then(async (rows) => {
      const msgs: Message[] = rows.map((r) => ({
        key: newKey(), role: r.role, content: r.content, runId: r.run_id ?? undefined,
        sources: [], trail: [], status: 'done',
      }));
      if (!alive) return;
      setMessages(msgs);
      const lastAssistant = [...msgs].reverse().find((m) => m.role === 'assistant');
      setFocused(lastAssistant?.key ?? null);
      const filled = await Promise.all(msgs.map(async (m) => {
        if (m.role !== 'assistant' || !m.runId) return m;
        const events = await api.pastEvents(m.runId).catch(() => [] as RunEvent[]);
        const rebuilt = events.reduce(applyEvent, { ...m, content: '' });
        return { ...rebuilt, content: m.content, status: 'done' as const };
      }));
      if (alive) setMessages(filled);
    }).catch(() => alive && setMessages([]));
    return () => { alive = false; };
  }, [session]);

  // ---------- actions ----------
  const updateMessage = (key: string, fn: (m: Message) => Message) =>
    setMessages((ms) => ms.map((m) => (m.key === key ? fn(m) : m)));

  const send = async (text: string) => {
    setNotice(null);
    let sid = session;
    if (!sid) {
      sid = (await api.newSession()).session_id;
      createdHere.current = sid;
      setSession(sid);
    }
    const userMsg: Message = { key: newKey(), role: 'user', content: text, sources: [], trail: [], status: 'done' };
    const key = newKey();
    const pending: Message = { key, role: 'assistant', content: '', sources: [], trail: [], status: 'streaming' };
    setMessages((ms) => [...ms, userMsg, pending]);
    setFocused(key);
    setActiveNode(null);
    setTab('sources');
    try {
      const { run_id } = await api.startRun(sid, text, scope);
      setRunId(run_id);
      updateMessage(key, (m) => ({ ...m, runId: run_id }));
      const es = new EventSource(`/api/runs/${run_id}/events`);
      stream.current = es;
      const onEvent = (e: MessageEvent) => {
        const ev: RunEvent = JSON.parse(e.data);
        updateMessage(key, (m) => applyEvent(m, ev));
        if (ev.type === 'done') {
          es.close();
          setRunId(null);
          loadSessions();
        }
      };
      EVENT_TYPES.forEach((t) => es.addEventListener(t, onEvent as EventListener));
    } catch (err) {
      updateMessage(key, (m) => ({ ...m, status: 'failed' }));
      setNotice((err as Error).message);
    }
  };

  const openSource = (s: Source) => {
    setReader({ paperId: s.paper_id, title: s.paper_title, page: s.page });
    setTab('reader');
    setShowPane(true);
  };

  const cite = (msgKey: string, nodeId: string) => {
    setFocused(msgKey);
    setActiveNode(nodeId);
    setTab('sources');
    setShowPane(true);
    requestAnimationFrame(() => document.getElementById(`src-${nodeId}`)?.scrollIntoView({ block: 'nearest' }));
  };

  const upload = async (files: File[]) => {
    try {
      const res = await api.upload(files);
      const rejected = res.papers.filter((p) => p.error);
      if (rejected.length) setNotice(`Skipped ${rejected.map((p) => `${p.filename} (${p.error})`).join(', ')}`);
    } catch (err) {
      setNotice(`Upload failed: ${(err as Error).message}`);
    }
    loadPapers();
  };

  const focusedMessage = useMemo(
    () => messages.find((m) => m.key === focused) ?? [...messages].reverse().find((m) => m.role === 'assistant') ?? null,
    [messages, focused]);
  const ready = papers.filter((p) => p.status === 'ready');
  const scopeTitles = ready.filter((p) => scope.includes(p.paper_id)).map((p) => p.title || p.filename || '');
  const modelReady = !!health?.ok && health.gen_present && health.embed_present;

  return (
    <div className="flex h-full flex-col">
      <header className="flex h-12 shrink-0 items-center gap-3 border-b border-rule bg-sheet px-4">
        <button className="shrink-0 text-[13px] text-ink-soft md:hidden" onClick={() => setShowLibrary((v) => !v)}>Library</button>
        <span className="whitespace-nowrap text-[15px] font-semibold">Paper Assistant</span>
        <ModelStatus health={health} unreachable={healthError} />
        <button className="ml-auto shrink-0 text-[13px] text-ink-soft xl:hidden" onClick={() => setShowPane((v) => !v)}>
          {showPane ? 'Hide sources' : 'Sources'}
        </button>
      </header>

      {notice && (
        <div role="status" className="flex items-center justify-between gap-3 border-b border-rule bg-mark-wash px-4 py-2 text-[13px]">
          <span>{notice}</span>
          <button className="text-xs font-medium" onClick={() => setNotice(null)}>Dismiss</button>
        </div>
      )}

      <div className="relative flex min-h-0 flex-1">
        <div className={`${showLibrary ? 'absolute inset-y-0 left-0 z-20 shadow-xl' : 'hidden'} w-[280px] shrink-0 md:static md:block md:shadow-none`}>
          <Library
            papers={papers}
            scope={scope}
            onToggleScope={(id) => setScope((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]))}
            onClearScope={() => setScope([])}
            onUpload={upload}
            onDelete={async (p) => {
              if (!confirm(`Remove “${p.title || p.filename}” and its index from this Mac?`)) return;
              await api.deletePaper(p.paper_id).catch((e) => setNotice(e.message));
              setScope((s) => s.filter((x) => x !== p.paper_id));
              loadPapers();
            }}
            onRetry={(p) => api.retryPaper(p.paper_id).then(loadPapers)}
            onOpenPaper={(p) => { setReader({ paperId: p.paper_id, title: p.title || p.filename || '', page: 1 }); setTab('reader'); setShowPane(true); }}
            sessions={sessions}
            currentSession={session}
            onSelectSession={(id) => { setSession(id); setShowLibrary(false); }}
            onNewChat={() => { setSession(null); setMessages([]); remember.set('pa.session', ''); setShowLibrary(false); }}
          />
        </div>

        <div className="min-w-0 flex-1">
          <Chat
            messages={messages}
            focused={focusedMessage?.key ?? null}
            activeNode={activeNode}
            running={!!runId}
            scopeTitles={scopeTitles}
            readyPapers={ready}
            modelReady={modelReady}
            onSend={send}
            onStop={() => runId && api.cancelRun(runId)}
            onFocus={(k) => { setFocused(k); setActiveNode(null); }}
            onCite={cite}
          />
        </div>

        <div className={`${showPane ? 'absolute inset-y-0 right-0 z-20 shadow-xl' : 'hidden'} w-[420px] max-w-full shrink-0 xl:static xl:block xl:shadow-none`}>
          <SidePane
            tab={tab}
            onTab={setTab}
            message={focusedMessage}
            activeNode={activeNode}
            onOpenSource={openSource}
            reader={reader}
            onClose={() => setShowPane(false)}
          />
        </div>
      </div>
    </div>
  );
}

function ModelStatus({ health, unreachable }: { health: Health | null; unreachable: boolean }) {
  if (unreachable) {
    return <span className="text-xs text-danger">The assistant's server isn't responding. Restart it with <code>uv run python -m app serve</code>.</span>;
  }
  if (!health) return <span className="text-xs text-ink-soft">Connecting…</span>;
  if (!health.ok) {
    return <span className="text-xs text-danger">Can't reach {health.runtime === 'ollama' ? 'Ollama' : 'the model server'}. Open the Ollama app; this page reconnects on its own.</span>;
  }
  const missing = [!health.gen_present && health.gen_model, !health.embed_present && health.embed_model].filter(Boolean);
  if (missing.length) {
    return <span className="text-xs text-danger">Model not downloaded. Run <code>ollama pull {missing.join(' && ollama pull ')}</code></span>;
  }
  return (
    <span className="flex min-w-0 items-center gap-2 text-xs text-ink-soft" title={`${health.gen_model}. Embeddings: ${health.embed_model}. Profile: ${health.profile}.`}>
      <span className="size-2 shrink-0 rounded-full bg-ok" aria-label="Model ready" />
      <span className="hidden truncate text-ink sm:inline">{health.gen_model}</span>
      <span className="hidden sm:inline">{Math.round(health.num_ctx / 1024)}k context</span>
      <span className="hidden lg:inline">about {health.memory_estimate_gb} of {health.gpu_budget_gb} GB</span>
      {health.ingest.queued > 0 && <span className="hidden sm:inline">reading {health.ingest.queued} {health.ingest.queued === 1 ? 'paper' : 'papers'}</span>}
    </span>
  );
}

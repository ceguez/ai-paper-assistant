import type { Health, Message, Paper, RunEvent, Session } from './types';

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

export const api = {
  health: () => fetch('/api/health').then((r) => json<Health>(r)),
  papers: () => fetch('/api/papers').then((r) => json<Paper[]>(r)),
  upload: (files: File[]) => {
    const form = new FormData();
    files.forEach((f) => form.append('files', f));
    return fetch('/api/papers', { method: 'POST', body: form })
      .then((r) => json<{ papers: { filename: string; status?: string; error?: string }[] }>(r));
  },
  deletePaper: (id: string) => fetch(`/api/papers/${id}`, { method: 'DELETE' }).then((r) => json(r)),
  retryPaper: (id: string) => fetch(`/api/papers/${id}/retry`, { method: 'POST' }).then((r) => json(r)),
  sessions: () => fetch('/api/sessions').then((r) => json<Session[]>(r)),
  newSession: () => fetch('/api/sessions', { method: 'POST' }).then((r) => json<{ session_id: string }>(r)),
  messages: (sid: string) =>
    fetch(`/api/sessions/${sid}/messages`).then((r) =>
      json<{ role: 'user' | 'assistant'; content: string; run_id: string | null }[]>(r)),
  startRun: (sid: string, message: string, paperIds: string[]) =>
    fetch(`/api/sessions/${sid}/runs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, paper_ids: paperIds }),
    }).then((r) => json<{ run_id: string }>(r)),
  cancelRun: (runId: string) => fetch(`/api/runs/${runId}/cancel`, { method: 'POST' }),
  /** Persisted events of a finished run (the server ends the stream after replay). */
  pastEvents: async (runId: string): Promise<RunEvent[]> => {
    const text = await fetch(`/api/runs/${runId}/events`).then((r) => (r.ok ? r.text() : ''));
    return text.split('\n').filter((l) => l.startsWith('data: ')).map((l) => JSON.parse(l.slice(6)));
  },
};

export const EVENT_TYPES = ['step', 'route', 'tool_call', 'tool_result', 'source', 'token', 'verify', 'error', 'done'];

/** Apply one run event to an assistant message. Pure, so live and replayed runs share it. */
export function applyEvent(msg: Message, ev: RunEvent): Message {
  const d = ev.data;
  switch (ev.type) {
    case 'token':
      return { ...msg, content: d.reset ? '' : msg.content + (d.delta ?? '') };
    case 'source':
      return msg.sources.some((s) => s.node_id === d.node_id)
        ? msg
        : { ...msg, sources: [...msg.sources, d as any], trail: [...msg.trail, ev] };
    case 'done':
      return { ...msg, content: d.answer || msg.content, status: d.status ?? 'done', trail: [...msg.trail, ev] };
    default:
      return { ...msg, trail: [...msg.trail, ev] };
  }
}

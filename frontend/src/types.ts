export type PaperStatus = 'queued' | 'processing' | 'ready' | 'failed';

export interface Paper {
  paper_id: string;
  title: string | null;
  authors: string | null;
  year: number | null;
  filename: string | null;
  status: PaperStatus;
  progress: string;
  error: string;
  page_count: number | null;
}

export interface Session {
  session_id: string;
  title: string;
  updated_at: number;
}

export interface Health {
  ok: boolean;
  runtime: string;
  profile: string;
  gen_model: string;
  embed_model: string;
  gen_present: boolean;
  embed_present: boolean;
  num_ctx: number;
  memory_estimate_gb: number;
  gpu_budget_gb: number;
  ingest: { current: string | null; queued: number };
}

export interface Source {
  node_id: string;
  paper_id: string;
  paper_title: string;
  breadcrumb: string;
  page: number | null;
  score: number;
  preview: string;
}

export type EventType =
  | 'step' | 'route' | 'tool_call' | 'tool_result' | 'source' | 'token' | 'verify' | 'error' | 'done';

export interface RunEvent {
  run_id: string;
  seq: number;
  ts: number;
  type: EventType;
  data: Record<string, any>;
}

export interface Message {
  key: string;
  role: 'user' | 'assistant';
  content: string;
  runId?: string;
  sources: Source[];
  trail: RunEvent[];
  status: 'streaming' | 'done' | 'cancelled' | 'failed';
}

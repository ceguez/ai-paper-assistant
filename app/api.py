"""FastAPI surface, with Swagger UI for testing in the browser.

Run:   uvicorn app.api:app --reload
Docs:  http://127.0.0.1:8000/docs  (Swagger UI)   http://127.0.0.1:8000/redoc  (ReDoc)
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field

from .agent.graph import Orchestrator
from .config import settings
from .db import Store
from .ingest.pipeline import fingerprint, ingest_pdf
from .llm import OllamaLLM
from .memory import after_run

DESCRIPTION = """
Local research assistant over your PDFs, running on a small model in Ollama.

**Test it from this page** (click an endpoint, then *Try it out*, then *Execute*):

1. `GET /health`: Ollama is reachable and both models are pulled.
2. `POST /papers`: choose a PDF. Copy the `paper_id` from the response.
3. `GET /papers`: wait until that paper's status is `ready` (parsing and embedding take a
   few seconds to a minute per paper).
4. `POST /sessions`: copy the `session_id`.
5. `POST /sessions/{session_id}/ask`: ask a question. The response holds the answer, its
   citations, the sections read and every step the agent took.
6. `GET /papers/{paper_id}`: browse the section tree that retrieval works over.

`GET /runs/{run_id}/events` is a live Server-Sent Events stream for the UI. Swagger shows it
only after the run ends; to watch it live use `curl -N http://127.0.0.1:8000/runs/<run_id>/events`.
"""

TAGS = [
    {"name": "System", "description": "Health of the model runtime."},
    {"name": "Papers", "description": "Upload PDFs and inspect what was indexed."},
    {"name": "Runs", "description": "Ask questions. Use `/ask` in Swagger; the SSE stream is for the UI."},
    {"name": "Memory", "description": "Facts the assistant has stored about your research."},
]

app = FastAPI(
    title="Research Paper Assistant",
    version="0.3.0",
    description=DESCRIPTION,
    openapi_tags=TAGS,
    swagger_ui_parameters={"tryItOutEnabled": True, "displayRequestDuration": True,
                           "defaultModelsExpandDepth": 0},
)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])

store = Store(settings.db_path)
llm = OllamaLLM()
agent = Orchestrator(store, llm, on_done=lambda s: after_run(store, llm, s.session_id, s.run_id, s.question, s.answer))
MAX_UPLOAD_MB = 100
pending_runs: dict[str, object] = {}  # run_id -> RunState waiting for its event stream


# ---------------- request and response schemas (shown in Swagger) ----------------
class RunRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000, description="Your question")

    model_config = {"json_schema_extra": {"examples": [
        {"message": "What optimizer and learning rate are used in training?"}]}}


class HealthOut(BaseModel):
    ollama: bool = Field(description="Ollama answered at the configured URL")
    gen_model: str
    gen_pulled: bool
    embed_model: str
    embed_pulled: bool
    num_ctx: int
    models: list[str] = Field(description="Every model Ollama reports as pulled")


class UploadOut(BaseModel):
    status: str = Field(description="`queued`, `already ingested` or `already processing`")
    paper_id: str
    file: str


class PaperOut(BaseModel):
    paper_id: str
    title: str | None = None
    authors: str | None = None
    year: int | None = None
    status: str | None = Field(None, description="`processing`, `ready` or `failed`")


class NodeOut(BaseModel):
    node_id: str
    parent_id: str | None = None
    kind: str = Field(description="`root`, `section`, `table` or `figure`")
    level: int
    title: str | None = None
    breadcrumb: str | None = None
    token_count: int | None = None


class SessionOut(BaseModel):
    session_id: str


class RunStarted(BaseModel):
    run_id: str


class SourceOut(BaseModel):
    node_id: str
    paper_id: str
    breadcrumb: str
    score: float = 0.0


class StepOut(BaseModel):
    seq: int
    type: str = Field(description="step, route, tool_call, tool_result, source, verify, error or done")
    data: dict[str, Any]


class AskOut(BaseModel):
    run_id: str
    status: str
    answer: str
    citations: list[str] = Field(description="node_ids cited in the answer as [n:<id>]")
    sources: list[SourceOut] = Field(description="Sections the agent read")
    steps: list[StepOut] = Field(description="Everything the agent did, without per-token events")


class RunOut(BaseModel):
    run_id: str
    status: str
    question: str
    route: str | None = None
    answer: str
    citations: list[str]
    errors: list[str]
    events: list[StepOut]


class FactOut(BaseModel):
    id: int
    text: str
    kind: str | None = None
    source_ref: str | None = None


class DeletedOut(BaseModel):
    deleted: int


def sse(ev: dict) -> str:
    return f"id: {ev['seq']}\nevent: {ev['type']}\ndata: {json.dumps(ev)}\n\n"


def _steps(events: list[dict]) -> list[dict]:
    return [{"seq": e["seq"], "type": e["type"], "data": e["data"]} for e in events if e["type"] != "token"]


# ---------------- system ----------------
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/docs")


@app.get("/health", tags=["System"], summary="Check Ollama and the configured models", response_model=HealthOut)
async def health():
    try:
        models = await llm.list_models()
    except Exception:
        models = None

    def pulled(name: str) -> bool:
        return bool(models) and (name in models or f"{name}:latest" in models)

    return {"ollama": models is not None, "models": models or [], "num_ctx": settings.budgets.num_ctx,
            "gen_model": settings.gen_model, "gen_pulled": pulled(settings.gen_model),
            "embed_model": settings.embed_model, "embed_pulled": pulled(settings.embed_model)}


# ---------------- papers ----------------
async def _ingest(path: str, paper_id: str) -> None:
    """Background ingestion that records failure, so a bad PDF shows `failed` instead of hanging."""
    try:
        await ingest_pdf(store, llm, path)
    except Exception as e:
        with store.tx() as c:
            c.execute("UPDATE papers SET status=? WHERE paper_id=? AND status!='ready'",
                      (f"failed: {str(e)[:200]}", paper_id))


@app.post("/papers", tags=["Papers"], summary="Upload a PDF and index it in the background",
          response_model=UploadOut, status_code=202,
          responses={400: {"description": "Not a PDF"}, 413: {"description": f"Larger than {MAX_UPLOAD_MB} MB"}})
async def upload_paper(file: UploadFile, background: BackgroundTasks):
    name = Path(file.filename or "paper.pdf").name
    dest = Path(settings.upload_dir) / f"{uuid.uuid4().hex}_{name}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    head = await file.read(5)
    if head != b"%PDF-":
        raise HTTPException(400, "not a PDF")
    size, limit = len(head), MAX_UPLOAD_MB * 1024 * 1024
    with dest.open("wb") as f:
        f.write(head)
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                f.close(); dest.unlink(missing_ok=True)
                raise HTTPException(413, f"larger than {MAX_UPLOAD_MB} MB")
            f.write(chunk)
    paper_id = fingerprint(str(dest))
    seen = store.conn.execute("SELECT status FROM papers WHERE paper_id=?", (paper_id,)).fetchone()
    if seen and seen["status"] in ("ready", "processing"):  # same bytes as before: keep the first copy
        dest.unlink(missing_ok=True)
        status = "already ingested" if seen["status"] == "ready" else "already processing"
        return {"status": status, "paper_id": paper_id, "file": name}
    with store.tx() as c:  # visible in GET /papers while it is being processed
        c.execute("INSERT OR REPLACE INTO papers(paper_id, title, file_path, status, created_at) "
                  "VALUES (?, ?, ?, 'processing', strftime('%s','now'))", (paper_id, name, str(dest)))
    background.add_task(_ingest, str(dest), paper_id)  # move to a job queue for multi-user
    return {"status": "queued", "paper_id": paper_id, "file": name}


@app.get("/papers", tags=["Papers"], summary="List papers and their indexing status",
         response_model=list[PaperOut])
async def list_papers():
    rows = store.conn.execute("SELECT paper_id, title, authors, year, status FROM papers ORDER BY created_at DESC")
    return [dict(r) for r in rows]


@app.get("/papers/{paper_id}", tags=["Papers"], summary="Section tree of a paper (what retrieval searches)",
         response_model=list[NodeOut], responses={404: {"description": "Unknown paper, or not indexed yet"}})
async def paper_tree(paper_id: str):
    rows = store.conn.execute("SELECT node_id, parent_id, kind, level, title, breadcrumb, token_count "
                              "FROM nodes WHERE paper_id=? ORDER BY level, ord", (paper_id,)).fetchall()
    if not rows:
        raise HTTPException(404, "unknown paper, or not indexed yet")
    return [dict(r) for r in rows]


@app.get("/papers/{paper_id}/file", tags=["Papers"], summary="Download the original PDF",
         response_class=FileResponse, responses={404: {"description": "Unknown paper"}})
async def paper_file(paper_id: str):
    row = store.conn.execute("SELECT file_path FROM papers WHERE paper_id=?", (paper_id,)).fetchone()
    if not row or not row["file_path"] or not Path(row["file_path"]).exists():
        raise HTTPException(404, "unknown paper")
    return FileResponse(row["file_path"], media_type="application/pdf")


# ---------------- sessions and runs ----------------
@app.post("/sessions", tags=["Runs"], summary="Start a chat session", response_model=SessionOut, status_code=201)
async def new_session():
    sid = uuid.uuid4().hex[:12]
    store.ensure_session(sid)
    return {"session_id": sid}


@app.post("/sessions/{session_id}/ask", tags=["Runs"], summary="Ask a question and wait for the full answer",
          response_model=AskOut)
async def ask(session_id: str, req: RunRequest):
    """Runs the agent to completion and returns the answer with its sources and steps.
    The best way to test from Swagger. Expect seconds to a minute on a laptop."""
    state = agent.new_run(session_id, req.message)
    events = [ev async for ev in agent.run(state)]
    done = next((e["data"] for e in reversed(events) if e["type"] == "done"), {})
    return {"run_id": state.run_id, "status": done.get("status", state.status),
            "answer": done.get("answer", state.answer), "citations": done.get("citations", []),
            "sources": [e["data"] for e in events if e["type"] == "source"], "steps": _steps(events)}


@app.post("/sessions/{session_id}/runs", tags=["Runs"], summary="Start a streamed run (for the UI)",
          response_model=RunStarted)
async def start_run(session_id: str, req: RunRequest):
    state = agent.new_run(session_id, req.message)
    pending_runs[state.run_id] = state
    return {"run_id": state.run_id}


@app.get("/runs/{run_id}", tags=["Runs"], summary="Result and recorded steps of a run", response_model=RunOut,
         responses={404: {"description": "Unknown run"}})
async def get_run(run_id: str):
    raw = store.load_run(run_id)
    if raw is None:
        raise HTTPException(404, "unknown run")
    s = json.loads(raw)
    return {"run_id": run_id, "status": s["status"], "question": s["question"], "route": s.get("route"),
            "answer": s.get("answer", ""), "citations": s.get("citations", []), "errors": s.get("errors", []),
            "events": _steps(store.events_since(run_id))}


@app.get("/runs/{run_id}/events", tags=["Runs"], summary="Server-Sent Events stream of a run",
         response_class=StreamingResponse,
         responses={200: {"content": {"text/event-stream": {}},
                          "description": "Live events; Swagger displays them only after the run ends"}})
async def run_events(run_id: str, request: Request):
    last = int(request.headers.get("last-event-id", "-1"))
    state = pending_runs.pop(run_id, None)

    async def stream():
        if state is None:  # reconnect or finished run: replay persisted events
            for ev in store.events_since(run_id, last):
                yield sse(ev)
            return
        async for ev in agent.run(state):
            if await request.is_disconnected():
                break
            yield sse(ev)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------- memory ----------------
@app.get("/memory/facts", tags=["Memory"], summary="Facts stored about your research", response_model=list[FactOut])
async def facts():
    rows = store.conn.execute("SELECT id, text, kind, source_ref FROM facts WHERE valid=1 ORDER BY id DESC")
    return [dict(r) for r in rows]


@app.delete("/memory/facts/{fact_id}", tags=["Memory"], summary="Delete a stored fact", response_model=DeletedOut)
async def delete_fact(fact_id: int):
    with store.tx() as c:
        c.execute("DELETE FROM facts WHERE id=?", (fact_id,))
    return {"deleted": fact_id}

"""FastAPI surface. The frontend only ever sees typed events over SSE (doc section 8).

Run:  uvicorn app.api:app --reload
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .agent.graph import Orchestrator
from .config import settings
from .db import Store
from .ingest.pipeline import ingest_pdf
from .llm import OllamaLLM
from .memory import after_run

app = FastAPI(title="Research Paper Assistant")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])

store = Store(settings.db_path)
llm = OllamaLLM()
agent = Orchestrator(store, llm, on_done=lambda s: after_run(store, llm, s.session_id, s.run_id, s.question, s.answer))
MAX_UPLOAD_MB = 100
pending_runs: dict[str, object] = {}  # run_id -> RunState waiting for its event stream


class RunRequest(BaseModel):
    message: str


def sse(ev: dict) -> str:
    return f"id: {ev['seq']}\nevent: {ev['type']}\ndata: {json.dumps(ev)}\n\n"


@app.post("/papers")
async def upload_paper(file: UploadFile, background: BackgroundTasks):
    dest = Path(settings.upload_dir) / f"{uuid.uuid4().hex}_{Path(file.filename or 'paper.pdf').name}"
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
    background.add_task(ingest_pdf, store, llm, str(dest))  # move to a job queue for multi-user
    return {"status": "queued", "file": dest.name}


@app.get("/papers")
async def list_papers():
    rows = store.conn.execute("SELECT paper_id, title, authors, year, status FROM papers ORDER BY created_at DESC")
    return [dict(r) for r in rows]


@app.get("/papers/{paper_id}")
async def paper_tree(paper_id: str):
    rows = store.conn.execute("SELECT node_id, parent_id, kind, level, title, breadcrumb, token_count "
                              "FROM nodes WHERE paper_id=? ORDER BY level, ord", (paper_id,)).fetchall()
    if not rows:
        raise HTTPException(404)
    return [dict(r) for r in rows]


@app.get("/papers/{paper_id}/file")
async def paper_file(paper_id: str):
    row = store.conn.execute("SELECT file_path FROM papers WHERE paper_id=?", (paper_id,)).fetchone()
    if not row or not row["file_path"]:
        raise HTTPException(404)
    return FileResponse(row["file_path"], media_type="application/pdf")


@app.post("/sessions")
async def new_session():
    sid = uuid.uuid4().hex[:12]
    store.ensure_session(sid)
    return {"session_id": sid}


@app.post("/sessions/{session_id}/runs")
async def start_run(session_id: str, req: RunRequest):
    state = agent.new_run(session_id, req.message)
    pending_runs[state.run_id] = state
    return {"run_id": state.run_id}


@app.get("/runs/{run_id}/events")
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


@app.get("/memory/facts")
async def facts():
    rows = store.conn.execute("SELECT id, text, kind, source_ref FROM facts WHERE valid=1 ORDER BY id DESC")
    return [dict(r) for r in rows]


@app.delete("/memory/facts/{fact_id}")
async def delete_fact(fact_id: int):
    with store.tx() as c:
        c.execute("DELETE FROM facts WHERE id=?", (fact_id,))
    return {"deleted": fact_id}

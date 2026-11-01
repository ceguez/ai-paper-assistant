"""HTTP API (Starlette) + static frontend. Everything under /api; the built UI is served at /.

Starlette is the ASGI layer FastAPI is built on. Routes here are plain and few, so the
extra layer buys little. Run with `python -m app serve` (binds 127.0.0.1 by default).
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from .agent.graph import Orchestrator
from .config import GPU_BUDGET_GB, ROOT, Settings, estimate_memory_gb, matches
from .config import settings as default_settings
from .db import Store
from .ingest.pipeline import fingerprint_bytes
from .jobs import IngestWorker
from .llm import LLM, make_llm
from .memory import after_run

log = logging.getLogger(__name__)
DIST = ROOT / "frontend" / "dist"


@dataclass
class Services:
    settings: Settings
    store: Store
    llm: LLM
    agent: Orchestrator
    worker: IngestWorker


async def probe_models(llm: LLM) -> list[str] | None:
    try:
        return await llm.list_models()
    except Exception:
        return None


async def build_services(settings: Settings | None = None, llm: LLM | None = None,
                         store: Store | None = None) -> Services:
    settings = settings or default_settings
    if llm is None:
        probe = make_llm(settings)
        available = await probe_models(probe)
        await probe.aclose()
        settings = settings.resolved(available)  # pick the best model you have actually pulled
        llm = make_llm(settings)
    store = store or Store(settings.db_path)
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)

    async def on_done(state):
        await after_run(store, llm, state.session_id, state.run_id, state.question, state.answer,
                        extract=settings.memory_extraction, history_budget=settings.budgets.history)

    agent = Orchestrator(store, llm, settings, on_done=on_done)
    return Services(settings, store, llm, agent, IngestWorker(store, llm))


def err(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def create_app(services: Services | None = None, *, start_worker: bool = True) -> Starlette:
    holder: dict[str, Services] = {}

    @contextlib.asynccontextmanager
    async def lifespan(app):
        svc = services or await build_services()
        holder["svc"] = svc
        recovered = svc.agent.recover_interrupted()
        if recovered:
            log.info("closed %d runs interrupted by the last shutdown", recovered)
        if start_worker:
            svc.worker.start()
        log.info("ready: runtime=%s model=%s embed=%s ctx=%d", svc.settings.runtime, svc.settings.gen_model,
                 svc.settings.embed_model, svc.settings.budgets.num_ctx)
        yield
        await svc.worker.stop()
        for task in list(svc.agent.tasks.values()):
            task.cancel()
        if services is None:
            await svc.llm.aclose()

    def S() -> Services:
        return holder["svc"]

    # ---------------- system ----------------
    async def health(request: Request):
        s = S()
        available = await probe_models(s.llm)
        st = s.settings
        papers = s.store.papers()
        return JSONResponse({
            "ok": available is not None,
            "runtime": st.runtime, "profile": st.profile,
            "gen_model": st.gen_model, "embed_model": st.embed_model,
            "gen_present": bool(available and matches(st.gen_model, available)),
            "embed_present": bool(available and matches(st.embed_model, available)),
            "models_available": available or [],
            "num_ctx": st.budgets.num_ctx,
            "memory_estimate_gb": estimate_memory_gb(st.gen_model, st.embed_model, st.budgets.num_ctx),
            "gpu_budget_gb": GPU_BUDGET_GB,
            "ingest": {"current": s.worker.current,
                       "queued": sum(p["status"] in ("queued", "processing") for p in papers)},
        })

    # ---------------- papers ----------------
    async def upload(request: Request):
        s = S()
        limit = s.settings.max_upload_mb * 1024 * 1024
        if int(request.headers.get("content-length") or 0) > limit * 20:
            return err(413, "upload too large")
        form = await request.form(max_files=20, max_part_size=limit)
        results = []
        for f in form.getlist("files"):
            if not hasattr(f, "read"):
                continue
            data = await f.read()
            name = Path(f.filename or "paper.pdf").name
            if len(data) > limit:
                results.append({"filename": name, "error": f"larger than {s.settings.max_upload_mb} MB"})
                continue
            if not data.startswith(b"%PDF-"):
                results.append({"filename": name, "error": "not a PDF"})
                continue
            paper_id = fingerprint_bytes(data)
            path = Path(s.settings.upload_dir) / f"{paper_id}.pdf"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(data)
            status = s.store.upsert_paper_stub(paper_id, name, str(path))
            results.append({"filename": name, "paper_id": paper_id, "status": status})
        await form.close()
        s.worker.notify()
        return JSONResponse({"papers": results}, status_code=202)

    async def list_papers(request: Request):
        return JSONResponse(S().store.papers())

    async def paper_detail(request: Request):
        s, pid = S(), request.path_params["paper_id"]
        row = s.store.paper(pid)
        if row is None:
            return err(404, "paper not found")
        return JSONResponse({"paper": {k: row[k] for k in row.keys() if k != "file_path"},
                             "tree": s.store.paper_tree(pid)})

    async def paper_file(request: Request):
        row = S().store.paper(request.path_params["paper_id"])
        if row is None or not row["file_path"] or not os.path.exists(row["file_path"]):
            return err(404, "file not found")
        return FileResponse(row["file_path"], media_type="application/pdf",
                            headers={"Content-Disposition": f'inline; filename="{row["filename"] or "paper.pdf"}"'})

    async def delete_paper(request: Request):
        s, pid = S(), request.path_params["paper_id"]
        if s.worker.current == pid:
            return err(409, "paper is being processed; try again when it finishes")
        path = s.store.delete_paper(pid)
        if path is None:
            return err(404, "paper not found")
        with contextlib.suppress(OSError):
            os.remove(path)
        return JSONResponse({"deleted": pid})

    async def retry_paper(request: Request):
        s, pid = S(), request.path_params["paper_id"]
        if s.store.paper(pid) is None:
            return err(404, "paper not found")
        s.store.set_paper_status(pid, "queued")
        s.worker.notify()
        return JSONResponse({"paper_id": pid, "status": "queued"})

    # ---------------- sessions and runs ----------------
    async def new_session(request: Request):
        sid = uuid.uuid4().hex[:12]
        S().store.ensure_session(sid)
        return JSONResponse({"session_id": sid}, status_code=201)

    async def list_sessions(request: Request):
        return JSONResponse(S().store.sessions())

    async def session_messages(request: Request):
        return JSONResponse(S().store.messages(request.path_params["session_id"]))

    async def start_run(request: Request):
        s, sid = S(), request.path_params["session_id"]
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return err(400, "invalid JSON")
        message = str(body.get("message", "")).strip()
        if not message:
            return err(400, "message is required")
        if len(message) > 4000:
            return err(400, "message too long (4000 characters max)")
        if active := s.agent.busy(sid):
            return err(409, f"run {active} is still in progress in this session")
        paper_ids = [p for p in body.get("paper_ids") or [] if isinstance(p, str)]
        state = s.agent.new_run(sid, message, paper_ids)
        s.agent.start(state)
        return JSONResponse({"run_id": state.run_id}, status_code=202)

    async def run_events(request: Request):
        s, run_id = S(), request.path_params["run_id"]
        if not s.agent.hub.has(run_id) and s.store.load_run(run_id) is None:
            return err(404, "run not found")
        after = int(request.headers.get("last-event-id") or request.query_params.get("after", "-1"))

        async def stream():
            yield "retry: 2000\n\n"
            async for ev in s.agent.events(run_id, after):
                yield f"id: {ev['seq']}\nevent: {ev['type']}\ndata: {json.dumps(ev)}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    async def cancel_run(request: Request):
        ok = S().agent.cancel(request.path_params["run_id"])
        return JSONResponse({"cancelled": ok}, status_code=200 if ok else 404)

    # ---------------- memory ----------------
    async def facts(request: Request):
        return JSONResponse(S().store.facts())

    async def delete_fact(request: Request):
        with S().store.tx() as c:
            c.execute("DELETE FROM facts WHERE id=?", (request.path_params["fact_id"],))
        return Response(status_code=204)

    async def index(request: Request):
        page = DIST / "index.html"
        if page.exists():
            return FileResponse(page)
        return HTMLResponse("<h1>Research Paper Assistant</h1><p>API is running. The UI is not built: "
                            "run <code>cd frontend && npm install && npm run build</code>.</p>", status_code=200)

    routes = [
        Route("/api/health", health),
        Route("/api/papers", upload, methods=["POST"]),
        Route("/api/papers", list_papers),
        Route("/api/papers/{paper_id}", paper_detail),
        Route("/api/papers/{paper_id}", delete_paper, methods=["DELETE"]),
        Route("/api/papers/{paper_id}/file", paper_file),
        Route("/api/papers/{paper_id}/retry", retry_paper, methods=["POST"]),
        Route("/api/sessions", new_session, methods=["POST"]),
        Route("/api/sessions", list_sessions),
        Route("/api/sessions/{session_id}/messages", session_messages),
        Route("/api/sessions/{session_id}/runs", start_run, methods=["POST"]),
        Route("/api/runs/{run_id}/events", run_events),
        Route("/api/runs/{run_id}/cancel", cancel_run, methods=["POST"]),
        Route("/api/memory/facts", facts),
        Route("/api/memory/facts/{fact_id:int}", delete_fact, methods=["DELETE"]),
        Route("/", index),
    ]
    if DIST.exists():
        routes.append(Mount("/", app=StaticFiles(directory=DIST)))
    return Starlette(routes=routes, lifespan=lifespan)


def app() -> Starlette:  # uvicorn --factory app.api:app
    return create_app()

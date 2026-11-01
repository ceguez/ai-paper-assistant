"""Command line: python -m app <command>

  serve            start the API + UI on http://127.0.0.1:8000
  doctor           check the runtime, models, memory fit and settings
  ingest PATH...   ingest PDFs (files or folders) without the UI
  ask "QUESTION"   ask from the terminal and stream the answer
  reindex          re-parse and re-embed every paper (after changing the embedding model)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from .config import GENERATORS, GPU_BUDGET_GB, PROFILES, estimate_memory_gb, matches, settings


def setup_logging(level: str = "INFO") -> None:
    logs = Path(settings.data_dir) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(logs / "app.log")])
    logging.getLogger("httpx").setLevel(logging.WARNING)


def ok(msg: str) -> None:
    print(f"  \033[32m✓\033[0m {msg}")


def bad(msg: str) -> None:
    print(f"  \033[31m✗\033[0m {msg}")


def note(msg: str) -> None:
    print(f"    {msg}")


async def doctor() -> int:
    from .api import probe_models
    from .llm import make_llm

    problems = 0
    print(f"Runtime: {settings.runtime}  profile: {settings.profile} ({PROFILES[settings.profile].description})")
    llm = make_llm(settings)
    available = await probe_models(llm)
    await llm.aclose()
    if available is None:
        bad(f"cannot reach {settings.runtime} at {settings.base_url or 'its default address'}")
        note("Start Ollama (open the Ollama app, or run `ollama serve`), then re-run `uv run python -m app doctor`.")
        return 1
    ok(f"{settings.runtime} is running ({len(available)} models pulled)")

    resolved = settings.resolved(available)
    gen, emb = resolved.gen_model, resolved.embed_model
    if matches(gen, available):
        ok(f"generator {gen} is pulled")
    else:
        problems += 1
        bad(f"generator {gen} is not pulled")
        note(f"Run: ollama pull {gen}")
    if matches(emb, available):
        ok(f"embedding model {emb} is pulled")
    else:
        problems += 1
        bad(f"embedding model {emb} is not pulled")
        note(f"Run: ollama pull {emb}")

    est = estimate_memory_gb(gen, emb, resolved.budgets.num_ctx)
    if est and est <= GPU_BUDGET_GB:
        ok(f"memory fit: ~{est} GB for models + KV cache at {resolved.budgets.num_ctx} context "
           f"(budget {GPU_BUDGET_GB} GB of 24 GB)")
    elif est:
        problems += 1
        bad(f"memory: ~{est} GB exceeds the {GPU_BUDGET_GB} GB budget; use PA_PROFILE=balanced or lower PA_NUM_CTX")
    if gen not in GENERATORS:
        note(f"{gen} is not in the sized model list; memory estimate skipped")

    if settings.runtime == "ollama":
        print("\nOllama server tuning (set where Ollama runs, then restart Ollama):")
        note("OLLAMA_FLASH_ATTENTION=1   OLLAMA_KV_CACHE_TYPE=q8_0   (halves KV cache memory)")
        note("OLLAMA_NUM_PARALLEL=1      OLLAMA_MAX_LOADED_MODELS=2  (generator + embedder stay resident)")
        note("For the Ollama Mac app: `bash scripts/tune_ollama.sh`, then quit and reopen Ollama.")

    try:  # parsers are optional extras; one is required
        import pymupdf4llm  # noqa: F401
        ok("PDF parser: pymupdf4llm")
    except ImportError:
        try:
            import docling  # noqa: F401
            ok("PDF parser: docling")
        except ImportError:
            problems += 1
            bad("no PDF parser installed: run `uv sync` (installs pymupdf4llm)")
    ok(f"data directory: {settings.data_dir}")
    print("\nAll checks passed." if not problems else f"\n{problems} problem(s) to fix.")
    return 1 if problems else 0


async def _services():
    from .api import build_services

    return await build_services()


async def ingest(paths: list[str]) -> int:
    from .ingest.pipeline import fingerprint

    svc = await _services()
    files = []
    for p in map(Path, paths):
        files += sorted(p.rglob("*.pdf")) if p.is_dir() else [p]
    for f in files:
        pid = fingerprint(str(f))
        dest = Path(svc.settings.upload_dir) / f"{pid}.pdf"
        if not dest.exists():
            dest.write_bytes(f.read_bytes())
        svc.store.upsert_paper_stub(pid, f.name, str(dest))
    pending = svc.store.pending_papers()
    for i, row in enumerate(pending, 1):
        print(f"[{i}/{len(pending)}] {Path(row['file_path']).name} ...", flush=True)
        await svc.worker.process(row["paper_id"], row["file_path"])
        status = svc.store.paper(row["paper_id"])
        print(f"    {status['status']}: {status['title']} {status['error'] or ''}")
    await svc.llm.aclose()
    return 0


async def ask(question: str, paper_ids: list[str]) -> int:
    svc = await _services()
    state = svc.agent.new_run("cli", question, paper_ids)
    async for ev in svc.agent.run(state):
        t, d = ev["type"], ev["data"]
        if t == "token":
            print(d.get("delta", ""), end="", flush=True)
        elif t == "route":
            print(f"\033[2m[route: {d['route']} | query: {d['query']}]\033[0m")
        elif t == "tool_call":
            print(f"\033[2m[{d['tool']} {d.get('query') or d.get('node_id', '')}]\033[0m")
        elif t == "source":
            print(f"\033[2m[source p.{d.get('page')}: {d['breadcrumb']}]\033[0m")
        elif t == "error":
            print(f"\033[33m[error in {d['node']}: {d['message']}]\033[0m")
        elif t == "done":
            print(f"\n\033[2m[{d['status']}; cited {len(d['citations'])} sources]\033[0m")
    await asyncio.sleep(0)  # let the post-run hook start
    await svc.llm.aclose()
    return 0


async def reindex() -> int:
    svc = await _services()
    for p in svc.store.papers():
        svc.store.set_paper_status(p["paper_id"], "queued")
    print(f"queued {len(svc.store.papers())} papers; they re-index when the server runs (or `python -m app ingest`)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--reload", action="store_true")
    sub.add_parser("doctor")
    i = sub.add_parser("ingest")
    i.add_argument("paths", nargs="+")
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--paper", action="append", default=[], help="limit to a paper_id (repeatable)")
    sub.add_parser("reindex")
    args = parser.parse_args(argv)

    if args.cmd == "serve":
        import uvicorn

        setup_logging()
        print(f"Research Paper Assistant on http://{settings.host}:{settings.port}")
        uvicorn.run("app.api:app", factory=True, host=settings.host, port=settings.port,
                    reload=args.reload, log_level="info")
        return 0
    setup_logging("WARNING")
    if args.cmd == "doctor":
        return asyncio.run(doctor())
    if args.cmd == "ingest":
        return asyncio.run(ingest(args.paths))
    if args.cmd == "ask":
        return asyncio.run(ask(args.question, args.paper))
    return asyncio.run(reindex())


if __name__ == "__main__":
    sys.exit(main())

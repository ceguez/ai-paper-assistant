"""Ingest a folder (or list) of PDFs without the server, timing each one.

    python scripts/ingest.py pdfDocs
    python scripts/ingest.py paper1.pdf paper2.pdf

Needs Ollama running with the configured models. Already-ingested papers are skipped.
Parse time is your CPU (or GPU with Docling); index time is mostly embedding calls to Ollama.
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from app.config import settings  # noqa: E402
from app.db import Store  # noqa: E402
from app.ingest.parser import parse_pdf  # noqa: E402
from app.ingest.pipeline import fingerprint, ingest_markdown  # noqa: E402
from app.llm import LLMError, OllamaLLM  # noqa: E402


async def main(args: list[str]) -> int:
    paths = [p for a in map(Path, args) for p in (sorted(a.glob("*.pdf")) if a.is_dir() else [a])]
    if not paths:
        print("usage: python scripts/ingest.py FOLDER_OR_PDF ...")
        return 2
    store, llm = Store(settings.db_path), OllamaLLM()
    print(f"{'paper':48} {'parser':12} {'parse s':>8} {'index s':>8} {'pointers':>8}")
    for path in paths:
        name = path.name[:47]
        paper_id = fingerprint(str(path))
        if store.paper_exists(paper_id):
            print(f"{name:48} already ingested")
            continue
        try:
            t0 = time.perf_counter()
            parsed = await asyncio.to_thread(parse_pdf, str(path))
            t1 = time.perf_counter()
            n = await ingest_markdown(store, llm, parsed.markdown, paper_id, str(path.resolve()))
            t2 = time.perf_counter()
            print(f"{name:48} {parsed.parser:12} {t1 - t0:8.1f} {t2 - t1:8.1f} {n:8d}")
        except LLMError as e:
            print(f"{name:48} model error: {e}\n  Is Ollama running with {settings.gen_model} and "
                  f"{settings.embed_model} pulled?")
            return 1
        except Exception as e:  # one bad PDF should not stop the batch
            print(f"{name:48} failed: {e}")
    print(f"database: {settings.db_path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))

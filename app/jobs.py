"""Persistent ingestion queue.

The papers table *is* the queue: an upload inserts a row with status 'queued', and one
worker processes rows oldest first. One at a time is deliberate on a single-GPU Mac:
parallel ingestion would fight chat for the same model and memory. On restart, papers left
'queued' or 'processing' are picked up again.
"""
from __future__ import annotations

import asyncio
import logging

from .db import Store
from .ingest.pipeline import ingest_pdf
from .llm import LLM

log = logging.getLogger(__name__)


class IngestWorker:
    def __init__(self, store: Store, llm: LLM):
        self.store, self.llm = store, llm
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        self.current: str | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="ingest-worker")
            self._wake.set()  # resume anything left over from the last run

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def notify(self) -> None:
        self._wake.set()

    async def _loop(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            while pending := self.store.pending_papers():
                await self.process(pending[0]["paper_id"], pending[0]["file_path"])

    async def process(self, paper_id: str, file_path: str) -> None:
        self.current = paper_id
        row = self.store.paper(paper_id)
        filename = row["filename"] if row else ""

        def progress(msg: str) -> None:
            self.store.set_paper_status(paper_id, "processing", msg)

        progress("starting")
        try:
            n = await ingest_pdf(self.store, self.llm, file_path, paper_id, filename, progress)
            log.info("ingested %s (%d pointers)", filename or paper_id, n)
        except asyncio.CancelledError:
            self.store.set_paper_status(paper_id, "queued", "interrupted")
            raise
        except Exception as e:
            log.exception("ingestion failed for %s", paper_id)
            self.store.set_paper_status(paper_id, "failed", "", str(e)[:500])
        finally:
            self.current = None

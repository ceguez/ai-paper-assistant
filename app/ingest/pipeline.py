"""Ingestion: parse -> metadata -> tree -> size -> pointers -> embed -> one atomic commit."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Awaitable, Callable

from pydantic import BaseModel, Field, field_validator

from ..db import Store, to_blob
from ..llm import LLM, LLMError
from .parser import ParsedDoc, parse_pdf
from .tree import build_tree, pointer_text, size_nodes, walk

log = logging.getLogger(__name__)
Progress = Callable[[str], Awaitable[None] | None]


class PaperMetadata(BaseModel):
    title: str = Field(max_length=300)
    authors: list[str] = Field(default_factory=list, max_length=50)
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None

    @field_validator("doi")
    @classmethod
    def _doi(cls, v):  # constrained decoding gives shape; regex gives truth
        return v if v and re.match(r"^10\.\d{4,9}/\S+$", v) else None

    @field_validator("arxiv_id")
    @classmethod
    def _arxiv(cls, v):
        return v if v and re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", v) else None

    @field_validator("year")
    @classmethod
    def _year(cls, v):
        return v if v and 1900 <= v <= 2100 else None


def fingerprint_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:24]


def fingerprint(path: str) -> str:
    return fingerprint_bytes(Path(path).read_bytes())


def _strip_markers(md: str) -> str:
    return re.sub(r"<!--\s*page:\d+\s*-->", "", md)


async def extract_metadata(llm: LLM, markdown: str, fallback_title: str) -> PaperMetadata:
    first_page = _strip_markers(markdown)[:3000]
    try:
        return await llm.structured(
            [{"role": "system", "content": "Extract bibliographic metadata from the start of a research paper. "
                                           "Use null when a field is absent. Do not guess."},
             {"role": "user", "content": first_page}],
            PaperMetadata, max_tokens=300)
    except LLMError as e:  # metadata is nice-to-have; never fail ingestion on it
        log.warning("metadata extraction failed: %s", e)
        h1 = re.search(r"^#\s+(.+)$", first_page, re.M)
        return PaperMetadata(title=(h1.group(1).strip() if h1 else fallback_title)[:300])


async def _report(progress: Progress | None, msg: str) -> None:
    if progress:
        res = progress(msg)
        if asyncio.iscoroutine(res):
            await res


async def ingest_markdown(store: Store, llm: LLM, markdown: str, paper_id: str, *, file_path: str = "",
                          filename: str = "", parser: str = "", page_count: int | None = None,
                          progress: Progress | None = None, batch: int = 32) -> int:
    """Everything after parsing. Separate so tests (and non-PDF sources) can call it directly."""
    await _report(progress, "reading metadata")
    meta = await extract_metadata(llm, markdown, Path(filename).stem or "Untitled paper")
    root = build_tree(markdown, paper_id, meta.title)
    size_nodes(root)
    nodes = list(walk(root))
    embeddable = [n for n in nodes if n.kind != "root" and n.embeddable and (n.text or n.kind != "section")]
    ptexts = [pointer_text(n) for n in embeddable]

    vectors: list[list[float]] = []
    for i in range(0, len(ptexts), batch):
        await _report(progress, f"embedding {min(i + batch, len(ptexts))}/{len(ptexts)}")
        vectors += await llm.embed(ptexts[i:i + batch])

    store.clear_paper_index(paper_id)  # re-ingest replaces the old tree
    now = time.time()
    with store.tx() as c:  # atomic: a crash leaves no half-indexed paper
        c.execute("INSERT INTO papers(paper_id, created_at) VALUES (?, ?) ON CONFLICT(paper_id) DO NOTHING",
                  (paper_id, now))
        c.execute("UPDATE papers SET title=?, authors=?, year=?, doi=?, arxiv_id=?, filename=COALESCE(NULLIF(?, ''), "
                  "filename), file_path=COALESCE(NULLIF(?, ''), file_path), status='ready', progress='', error='', "
                  "parser=?, page_count=?, node_count=?, updated_at=? WHERE paper_id=?",
                  (meta.title, ", ".join(meta.authors), meta.year, meta.doi, meta.arxiv_id, filename, file_path,
                   parser, page_count, len(nodes), now, paper_id))
        for n in nodes:
            c.execute("INSERT OR REPLACE INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (n.node_id, paper_id, n.parent_id, n.kind, n.level, n.ord, n.title, n.breadcrumb,
                       n.text, n.token_count, n.page_start, n.page_end, int(n.embeddable)))
        for n, text, vec in zip(embeddable, ptexts, vectors):
            cur = c.execute("INSERT INTO pointers(node_id, paper_id, text, embedding, embed_model, dim) "
                            "VALUES (?,?,?,?,?,?)", (n.node_id, paper_id, text, to_blob(vec),
                                                     llm.settings.embed_model, len(vec)))
            c.execute("INSERT INTO pointers_fts(rowid, text) VALUES (?,?)", (cur.lastrowid, text))
    store.vectors.dirty = True
    return len(embeddable)


async def ingest_pdf(store: Store, llm: LLM, path: str, paper_id: str, filename: str = "",
                     progress: Progress | None = None) -> int:
    await _report(progress, "parsing PDF")
    # Parsing is CPU-bound (and Docling loads models): keep it off the event loop.
    parsed: ParsedDoc = await asyncio.to_thread(parse_pdf, path, llm.settings.parser)
    return await ingest_markdown(store, llm, parsed.markdown, paper_id, file_path=path,
                                 filename=filename or Path(path).name, parser=parsed.parser,
                                 page_count=parsed.page_count, progress=progress)

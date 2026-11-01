"""Ingestion: fingerprint -> parse -> metadata -> tree -> size -> pointers -> embed -> one atomic commit."""
from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from ..db import Store, to_blob
from ..llm import LLM
from ..config import settings
from .parser import parse_pdf
from .tree import build_tree, pointer_text, size_nodes, walk


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


def fingerprint(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:24]


async def extract_metadata(llm: LLM, markdown: str) -> PaperMetadata:
    first_page = markdown[:3000]
    return await llm.structured(
        [{"role": "system", "content": "Extract bibliographic metadata from the start of a research paper. "
                                       "Use null when a field is absent. Do not guess."},
         {"role": "user", "content": first_page}],
        PaperMetadata, max_tokens=300)


async def ingest_markdown(store: Store, llm: LLM, markdown: str, paper_id: str,
                          file_path: str = "", batch: int = 32) -> int:
    """Everything after parsing. Separate so tests (and non-PDF sources) can call it directly."""
    meta = await extract_metadata(llm, markdown)
    root = build_tree(markdown, paper_id, meta.title)
    size_nodes(root)
    nodes = list(walk(root))
    embeddable = [n for n in nodes if n.kind != "root" and n.embeddable and (n.text or n.kind != "section")]
    ptexts = [pointer_text(n) for n in embeddable]

    vectors: list[list[float]] = []
    for i in range(0, len(ptexts), batch):
        vectors += await llm.embed(ptexts[i:i + batch])

    with store.tx() as c:  # atomic: a crash leaves no half-indexed paper
        c.execute("INSERT OR REPLACE INTO papers VALUES (?,?,?,?,?,?,?,?,?)",
                  (paper_id, meta.title, ", ".join(meta.authors), meta.year, meta.doi, meta.arxiv_id,
                   file_path, "ready", time.time()))
        for n in nodes:
            c.execute("INSERT OR REPLACE INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (n.node_id, paper_id, n.parent_id, n.kind, n.level, n.ord, n.title, n.breadcrumb,
                       n.text, n.token_count, None, None, int(n.embeddable)))
        for n, text, vec in zip(embeddable, ptexts, vectors):
            cur = c.execute("INSERT INTO pointers(node_id, paper_id, text, embedding, embed_model, dim) "
                            "VALUES (?,?,?,?,?,?)", (n.node_id, paper_id, text, to_blob(vec),
                                                     settings.embed_model, len(vec)))
            c.execute("INSERT INTO pointers_fts(rowid, text) VALUES (?,?)", (cur.lastrowid, text))
    return len(embeddable)


async def ingest_pdf(store: Store, llm: LLM, path: str) -> str:
    paper_id = fingerprint(path)
    if store.paper_exists(paper_id):
        return paper_id  # idempotent re-upload
    parsed = parse_pdf(path)
    await ingest_markdown(store, llm, parsed.markdown, paper_id, path)
    return paper_id

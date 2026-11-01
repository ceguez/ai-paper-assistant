"""Post-run memory maintenance, off the hot path.

- compact_session: when history outgrows its slot, fold the oldest turns into the rolling summary.
- extract_facts: distil durable semantic facts from a finished run, deduplicated against
  what is already stored, each with provenance so the UI can show and delete it.
- record_episode: one-line episodic record of what the run did.
"""
from __future__ import annotations

import time

from pydantic import BaseModel, Field

from .db import Store, to_blob
from .ingest.tree import count_tokens
from .llm import LLM


class SessionSummary(BaseModel):
    topics: list[str] = Field(default_factory=list, max_length=8)
    open_questions: list[str] = Field(default_factory=list, max_length=5)
    decisions: list[str] = Field(default_factory=list, max_length=5)
    papers_referenced: list[str] = Field(default_factory=list, max_length=10)

    def render(self) -> str:
        return "; ".join(f"{k}: {', '.join(v)}" for k, v in self.model_dump().items() if v)


class FactOp(BaseModel):
    op: str = Field(pattern="^(add|keep|invalidate)$")
    text: str = Field(max_length=300)
    kind: str = Field(default="research_interest", pattern="^(research_interest|preference|paper_claim)$")
    existing_id: int | None = None


class FactUpdate(BaseModel):
    ops: list[FactOp] = Field(default_factory=list, max_length=6)


async def compact_session(store: Store, llm: LLM, session_id: str, history_budget: int = 1500) -> bool:
    turns = store.history(session_id, limit=200)
    if sum(count_tokens(t["content"]) for t in turns) <= history_budget:
        return False
    old = turns[: len(turns) // 2]
    prior = store.session_summary(session_id)
    transcript = "\n".join(f"{t['role']}: {t['content']}" for t in old)
    summary = await llm.structured(
        [{"role": "system", "content": "Update the running summary of a research conversation."},
         {"role": "user", "content": f"<prior>{prior}</prior>\n<turns>{transcript[:12000]}</turns>"}],
        SessionSummary, max_tokens=400)
    with store.tx() as c:
        c.execute("UPDATE sessions SET summary=? WHERE session_id=?", (summary.render(), session_id))
        # Turns stay in `messages` for audit; ContextAssembler only reads the recent window.
    return True


async def extract_facts(store: Store, llm: LLM, session_id: str, run_id: str, question: str, answer: str) -> int:
    existing = store.conn.execute("SELECT id, text FROM facts WHERE valid=1 ORDER BY id DESC LIMIT 30").fetchall()
    known = "\n".join(f"{r['id']}: {r['text']}" for r in existing) or "none"
    update = await llm.structured(
        [{"role": "system", "content": "Maintain durable facts about the user's research. Only add facts the user "
                                       "stated or that the cited answer establishes. Invalidate facts contradicted now."},
         {"role": "user", "content": f"<known>{known}</known>\n<question>{question}</question>\n<answer>{answer[:3000]}</answer>"}],
        FactUpdate, max_tokens=400)
    changed = 0
    adds = [o for o in update.ops if o.op == "add"]
    vecs = await llm.embed([o.text for o in adds]) if adds else []
    now = time.time()
    with store.tx() as c:
        for op, vec in zip(adds, vecs):
            c.execute("INSERT INTO facts(text, kind, source_ref, embedding, created_at, updated_at) VALUES (?,?,?,?,?,?)",
                      (op.text, op.kind, f"run:{run_id}", to_blob(vec), now, now))
            changed += 1
        for op in update.ops:
            if op.op == "invalidate" and op.existing_id:
                c.execute("UPDATE facts SET valid=0, updated_at=? WHERE id=?", (now, op.existing_id))
                changed += 1
    return changed


async def record_episode(store: Store, llm: LLM, session_id: str, run_id: str, question: str, answer: str) -> None:
    summary = f"Q: {question[:200]} | A: {answer[:300]}"
    vec = (await llm.embed([summary]))[0]
    with store.tx() as c:
        c.execute("INSERT INTO episodes(session_id, run_id, summary, embedding, created_at) VALUES (?,?,?,?,?)",
                  (session_id, run_id, summary, to_blob(vec), time.time()))


async def after_run(store: Store, llm: LLM, session_id: str, run_id: str, question: str, answer: str) -> None:
    """Called as a background task once a run is done. Failures here never affect the user."""
    for job in (record_episode(store, llm, session_id, run_id, question, answer),
                extract_facts(store, llm, session_id, run_id, question, answer),
                compact_session(store, llm, session_id)):
        try:
            await job
        except Exception:
            pass  # log in production

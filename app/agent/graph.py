"""Native orchestration: an explicit graph of async nodes over one typed RunState.

    route ──chat──────────────────────────────────────────────► answer ─► verify ─► done
      └─search/deep_read/compare─► retrieve ─► research (tool loop, capped) ─┘
                                        └── search route skips research ─────┘
    any node failure ─► fallback (plain retrieval + one answer call)

The model picks the route and, inside `research`, the next tool. Code owns every
transition, the step cap, loop detection and fallbacks. State is checkpointed after
every node.

Runs execute as background tasks and publish to a RunHub. Clients subscribe (and
re-subscribe after a refresh) from any sequence number, so closing the browser tab
does not kill a run; only an explicit cancel does.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import OrderedDict
from typing import AsyncIterator, Awaitable, Callable

from ..config import Settings
from ..db import Store
from ..llm import LLM, LLMError
from .context import ContextAssembler
from .state import Route, RouteDecision, RunState, ToolCall, Verification
from .tools import Tools

log = logging.getLogger(__name__)
Event = dict
NodeFn = Callable[[RunState], Awaitable[str]]
DoneHook = Callable[[RunState], Awaitable[None]]

ROUTER_INSTRUCTIONS = (
    "Classify the user's request. chat: greetings, thanks, or questions about you. search: find which papers or "
    "sections discuss a topic. deep_read: answer a specific question from paper content. compare: contrast "
    "two or more papers. Also write a keyword-rich search_query."
)
RESEARCH_INSTRUCTIONS = (
    "Decide the single next action to gather evidence. Tools: search_papers(query) finds sections; "
    "read_section(node_id) loads a full section into sources; get_element(node_id) loads a table or figure; "
    "save_note(claim, node_id) records a finding from a section already in sources; finish when the sources "
    "suffice to answer. Never repeat an action listed in progress. Prefer finish over guessing."
)
ANSWER_INSTRUCTIONS = ("Answer the question from the sources, citing ids inline like [n:abc123]. "
                       "Use Markdown. If the sources are insufficient, say what is missing.")
SEARCH_INSTRUCTIONS = ("List which papers and sections in the sources address the question, one bullet each with a "
                       "one-line reason and its id like [n:abc123].")
CHAT_INSTRUCTIONS = "Reply briefly and helpfully. You help the user read and compare research papers in their library."
VERIFY_INSTRUCTIONS = "List each factual claim in the draft answer and whether the cited sources support it."


class RunHub:
    """In-memory event log per run with async fan-out to any number of subscribers."""

    def __init__(self, keep: int = 100):
        self.logs: OrderedDict[str, list[Event]] = OrderedDict()
        self.conds: dict[str, asyncio.Condition] = {}
        self.finished: set[str] = set()
        self.keep = keep

    def open(self, run_id: str, first_seq: int = 0) -> None:
        self.logs[run_id] = []
        self.conds[run_id] = asyncio.Condition()
        while len(self.logs) > self.keep:  # evict the oldest finished run
            old = next((r for r in self.logs if r in self.finished), None)
            if old is None:
                break
            self.logs.pop(old)
            self.conds.pop(old, None)
            self.finished.discard(old)

    async def publish(self, ev: Event) -> None:
        run_id = ev["run_id"]
        if run_id not in self.logs:
            return
        self.logs[run_id].append(ev)
        async with self.conds[run_id]:
            self.conds[run_id].notify_all()

    async def close(self, run_id: str) -> None:
        self.finished.add(run_id)
        if cond := self.conds.get(run_id):
            async with cond:
                cond.notify_all()

    def has(self, run_id: str) -> bool:
        return run_id in self.logs

    async def subscribe(self, run_id: str, after_seq: int = -1) -> AsyncIterator[Event]:
        log_, cond = self.logs[run_id], self.conds[run_id]
        while True:
            for ev in [e for e in log_ if e["seq"] > after_seq]:
                after_seq = ev["seq"]
                yield ev
            if run_id in self.finished:
                return
            async with cond:
                if not any(e["seq"] > after_seq for e in log_) and run_id not in self.finished:
                    await cond.wait()


class Orchestrator:
    def __init__(self, store: Store, llm: LLM, settings: Settings, hub: RunHub | None = None,
                 on_done: DoneHook | None = None):
        self.store, self.llm, self.settings = store, llm, settings
        self.ctx = ContextAssembler(settings.budgets)
        self.tools = Tools(store, llm)
        self.hub = hub or RunHub()
        self.on_done = on_done
        self.nodes: dict[str, NodeFn] = {
            "route": self.route, "retrieve": self.retrieve, "research": self.research,
            "answer": self.answer, "verify": self.verify, "fallback": self.fallback,
        }
        self._seq: dict[str, int] = {}
        self.tasks: dict[str, asyncio.Task] = {}
        self.active_by_session: dict[str, str] = {}
        self._titles: dict[str, str] = {}

    # ---------------- lifecycle ----------------
    def new_run(self, session_id: str, question: str, paper_ids: list[str] | None = None) -> RunState:
        self.store.ensure_session(session_id, title=question[:80])
        state = RunState(run_id=uuid.uuid4().hex[:12], session_id=session_id, question=question,
                         paper_ids=paper_ids or [], max_steps=self.settings.max_steps)
        self.checkpoint(state)
        return state

    def busy(self, session_id: str) -> str | None:
        run_id = self.active_by_session.get(session_id)
        return run_id if run_id and run_id in self.tasks and not self.tasks[run_id].done() else None

    def start(self, state: RunState) -> asyncio.Task:
        self.hub.open(state.run_id)
        self._seq[state.run_id] = self.store.last_seq(state.run_id)
        task = asyncio.create_task(self._drive(state), name=f"run-{state.run_id}")
        self.tasks[state.run_id] = task
        self.active_by_session[state.session_id] = state.run_id
        task.add_done_callback(lambda t, rid=state.run_id: self.tasks.pop(rid, None))
        return task

    def cancel(self, run_id: str) -> bool:
        task = self.tasks.get(run_id)
        if task and not task.done():
            task.cancel()
            return True
        return False

    async def run(self, state: RunState) -> AsyncIterator[Event]:
        """Start a run and stream its events (CLI and tests)."""
        self.start(state)
        async for ev in self.hub.subscribe(state.run_id):
            yield ev

    async def events(self, run_id: str, after_seq: int = -1) -> AsyncIterator[Event]:
        """Live events if the run is in memory, otherwise replay what was persisted."""
        if self.hub.has(run_id):
            async for ev in self.hub.subscribe(run_id, after_seq):
                yield ev
        else:
            for ev in self.store.events_since(run_id, after_seq):
                yield ev

    def recover_interrupted(self) -> int:
        """Runs still 'running' in the DB were killed by a restart: close them out."""
        ids = self.store.interrupted_runs()
        for run_id in ids:
            raw = self.store.load_run(run_id)
            if not raw:
                continue
            state = RunState.model_validate_json(raw)
            state.status = "failed"
            state.errors.append("interrupted by server restart")
            self.store.checkpoint(run_id, state.session_id, state.model_dump_json(), state.status)
            seq = self.store.last_seq(run_id) + 1
            self.store.add_event(run_id, seq, "done", {"answer": state.answer, "citations": state.citations,
                                                       "status": "failed", "error": "interrupted by server restart"})
        return len(ids)

    # ---------------- plumbing ----------------
    async def emit(self, state: RunState, type_: str, data: dict) -> Event:
        seq = self._seq.get(state.run_id, -1) + 1
        self._seq[state.run_id] = seq
        if type_ != "token":  # persist everything but per-token deltas (`done` carries the full answer)
            self.store.add_event(state.run_id, seq, type_, data)
        ev = {"run_id": state.run_id, "seq": seq, "ts": time.time(), "type": type_, "data": data}
        await self.hub.publish(ev)
        return ev

    def checkpoint(self, state: RunState) -> None:
        self.store.checkpoint(state.run_id, state.session_id, state.model_dump_json(), state.status)

    def _paper_title(self, paper_id: str) -> str:
        if paper_id not in self._titles:
            row = self.store.paper(paper_id)
            self._titles[paper_id] = (row["title"] or row["filename"]) if row else paper_id
        return self._titles[paper_id]

    async def _emit_source(self, state: RunState, node_id: str) -> None:
        ev = state.evidence[node_id]
        await self.emit(state, "source", {"node_id": node_id, "paper_id": ev.paper_id,
                                          "paper_title": self._paper_title(ev.paper_id),
                                          "breadcrumb": ev.breadcrumb, "page": ev.page,
                                          "score": round(ev.score, 4), "preview": ev.text[:280]})

    async def _drive(self, state: RunState) -> None:
        """Execute nodes from state.node until done. Resumable: state.node is the resume point."""
        if state.node == "route":
            self.store.add_message(state.session_id, "user", state.question, state.run_id)
        try:
            while state.node != "done" and state.status == "running":
                current = state.node
                await self.emit(state, "step", {"node": current, "phase": "start"})
                t0 = time.perf_counter()
                try:
                    state.node = await self.nodes[current](state)
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # any node failure -> deterministic fallback, once
                    log.warning("node %s failed: %s", current, e)
                    state.errors.append(f"{current}: {e}")
                    await self.emit(state, "error", {"node": current, "message": str(e)[:300]})
                    if current == "fallback":
                        state.status = "failed"
                        break
                    state.node = "fallback"
                await self.emit(state, "step", {"node": current, "phase": "end",
                                                "ms": int((time.perf_counter() - t0) * 1000)})
                self.checkpoint(state)
        except asyncio.CancelledError:
            state.status = "cancelled"
        if state.status == "running":
            state.status = "done"
        if state.answer:
            self.store.add_message(state.session_id, "assistant", state.answer, state.run_id)
        await self.emit(state, "done", {"answer": state.answer, "citations": state.citations,
                                        "status": state.status, "errors": state.errors[-3:]})
        self.checkpoint(state)
        await self.hub.close(state.run_id)
        if self.on_done and state.status == "done":
            asyncio.create_task(self._safe_hook(state))

    async def _safe_hook(self, state: RunState) -> None:
        try:
            await self.on_done(state)  # type: ignore[misc]
        except Exception:
            log.exception("post-run hook failed")

    # ---------------- nodes ----------------
    async def route(self, state: RunState) -> str:
        msgs = self.ctx.build(instructions=ROUTER_INSTRUCTIONS, question=state.question,
                              turns=self.store.history(state.session_id, 6)[:-1])
        decision = await self.llm.structured(msgs, RouteDecision, model=self.settings.router, max_tokens=200)
        state.route, state.search_query = decision.route, decision.search_query or state.question
        await self.emit(state, "route", {"route": decision.route.value, "query": state.search_query,
                                         "reasoning": decision.reasoning})
        return "answer" if decision.route == Route.chat else "retrieve"

    async def retrieve(self, state: RunState) -> str:
        """Deterministic seed retrieval: cheap, reliable, and gives the loop a head start."""
        await self.emit(state, "tool_call", {"tool": "search_papers", "query": state.search_query,
                                             "reasoning": "seed retrieval"})
        result = await self.tools.search_papers(state, state.search_query)
        await self.emit(state, "tool_result", {"tool": "search_papers", "summary": result[:600]})
        n_read = {Route.compare: 2, Route.search: 3}.get(state.route, 1)  # type: ignore[arg-type]
        for cand in state.candidates[:n_read]:
            self.tools.read_section(state, cand["node_id"])
        for nid in state.evidence:
            await self._emit_source(state, nid)
        return "answer" if state.route == Route.search else "research"

    async def research(self, state: RunState) -> str:
        """Bounded tool loop. The SLM chooses actions; code enforces caps and dedupe."""
        seen = {s["sig"] for s in state.scratchpad}
        while state.step < state.max_steps:
            state.step += 1
            progress = "\n".join(f"{s['tool']}({s['arg']}) -> {s['result'][:160]}" for s in state.scratchpad[-6:])
            unread = [c for c in state.candidates if c["node_id"] not in state.evidence][:6]
            cands = "\n".join(f"{c['node_id']} | {c['breadcrumb']}" for c in unread)
            msgs = self.ctx.build(instructions=RESEARCH_INSTRUCTIONS, question=state.question,
                                  evidence=list(state.evidence.values()),
                                  scratchpad=f"{progress}\nunread candidates:\n{cands}".strip())
            call = await self.llm.structured(msgs, ToolCall, temperature=0.0, max_tokens=250)
            sig = state.call_signature(call)
            if call.tool == "finish":
                break
            if sig in seen:  # small models loop; a repeat means it has nothing better to do
                await self.emit(state, "step", {"node": "research", "phase": "note",
                                                "note": "repeated action; finishing"})
                break
            seen.add(sig)
            await self.emit(state, "tool_call", {"tool": call.tool, "query": call.query, "node_id": call.node_id,
                                                 "claim": call.claim, "reasoning": call.reasoning})
            before = set(state.evidence)
            result = await self.tools.dispatch(state, call)
            state.scratchpad.append({"sig": sig, "tool": call.tool, "arg": call.query or call.node_id,
                                     "result": result})
            await self.emit(state, "tool_result", {"tool": call.tool, "summary": result[:600]})
            for nid in set(state.evidence) - before:
                await self._emit_source(state, nid)
            self.checkpoint(state)
        return "answer"

    async def answer(self, state: RunState) -> str:
        chat = state.route == Route.chat
        instructions = CHAT_INSTRUCTIONS if chat else (
            SEARCH_INSTRUCTIONS if state.route == Route.search else ANSWER_INSTRUCTIONS)
        msgs = self.ctx.build(instructions=instructions, question=state.question,
                              evidence=None if chat else list(state.evidence.values()),
                              summary=self.store.session_summary(state.session_id),
                              turns=self.store.history(state.session_id, 10)[:-1],
                              facts=self.store.top_facts(None), notes=self.store.notes(state.session_id))
        parts: list[str] = []
        async for delta in self.llm.stream(msgs, temperature=0.3 if not chat else 0.6,
                                           max_tokens=self.settings.budgets.output_reserve):
            parts.append(delta)
            await self.emit(state, "token", {"delta": delta})
        state.answer = "".join(parts).strip()
        state.citations = [nid for nid in state.evidence if f"n:{nid}" in state.answer]
        return "verify" if state.evidence and state.route in (Route.deep_read, Route.compare) else "done"

    async def verify(self, state: RunState) -> str:
        msgs = self.ctx.build(instructions=VERIFY_INSTRUCTIONS, evidence=list(state.evidence.values()),
                              question=f"Draft answer to check:\n{state.answer}")
        try:
            result = await self.llm.structured(msgs, Verification, max_tokens=600)
        except LLMError as e:  # verification is best-effort; never block the answer on it
            await self.emit(state, "verify", {"skipped": True, "reason": str(e)[:200]})
            return "done"
        unsupported = [c.claim for c in result.checks if not c.supported]
        if unsupported:
            addendum = "\n\n> **Not verified against the sources:** " + "; ".join(unsupported)
            state.answer += addendum
            await self.emit(state, "token", {"delta": addendum})
        await self.emit(state, "verify", {"checked": len(result.checks), "unsupported": unsupported})
        return "done"

    async def fallback(self, state: RunState) -> str:
        """Deterministic path that needs no structured output from the model."""
        if not state.evidence:
            qvec = (await self.llm.embed([state.question]))[0]
            hits = self.store.search(state.question, qvec, k=3, paper_ids=state.paper_ids or None)
            state.candidates = [h.__dict__ for h in hits]
            for h in hits[:2]:
                self.tools.read_section(state, h.node_id)
            for nid in state.evidence:
                await self._emit_source(state, nid)
        if state.route in (None, Route.chat) and state.evidence:
            state.route = Route.deep_read
        state.route = state.route or Route.deep_read
        await self.emit(state, "token", {"delta": "", "reset": True})  # discard any partial answer in the UI
        await self.answer(state)
        return "done"

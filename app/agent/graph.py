"""Native orchestration: an explicit graph of async nodes over one typed RunState.

    route ──chat──────────────────────────────► answer ─► verify ─► done
      └─search/deep_read/compare─► retrieve ─► research (tool loop, capped) ─┘
                                        └── search route skips research ──┘
    any node failure ─► fallback (plain retrieval + one answer call)

The model picks the route and, inside `research`, the next tool. Code owns every
transition, the step cap, loop detection and fallbacks. State is checkpointed after
every node, so runs are resumable and the UI can replay the trace.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import AsyncIterator, Awaitable, Callable

from ..config import settings
from ..db import Store
from ..llm import LLM, LLMError
from .context import ContextAssembler
from .state import Evidence, Route, RouteDecision, RunState, ToolCall, Verification
from .tools import Tools

Event = dict
NodeFn = Callable[[RunState], Awaitable[str]]

ROUTER_INSTRUCTIONS = (
    "Classify the user's request. chat: greetings or questions about you. search: find which papers or "
    "sections discuss a topic. deep_read: answer a specific question from paper content. compare: contrast "
    "two or more papers. Also write a keyword-rich search_query."
)
RESEARCH_INSTRUCTIONS = (
    "Decide the single next action to gather evidence. Tools: search_papers(query) finds sections; "
    "read_section(node_id) loads a full section into sources; get_element(node_id) loads a table or figure; "
    "save_note(claim, node_id) records a finding from a section already in sources; finish when the sources "
    "suffice to answer. Never repeat an action listed in progress. Prefer finish over guessing."
)
ANSWER_INSTRUCTIONS = "Answer the question from the sources, citing ids inline. If sources are insufficient, say what is missing."
VERIFY_INSTRUCTIONS = "List each factual claim in the draft answer and whether the cited sources support it."


class Orchestrator:
    def __init__(self, store: Store, llm: LLM, assembler: ContextAssembler | None = None):
        self.store, self.llm = store, llm
        self.ctx = assembler or ContextAssembler()
        self.tools = Tools(store, llm)
        self.nodes: dict[str, NodeFn] = {
            "route": self.route, "retrieve": self.retrieve, "research": self.research,
            "answer": self.answer, "verify": self.verify, "fallback": self.fallback,
        }
        self._seq: dict[str, int] = {}
        self._queues: dict[str, asyncio.Queue] = {}  # one live event queue per run

    # ---------------- plumbing ----------------
    def emit(self, state: RunState, type_: str, data: dict) -> Event:
        seq = self._seq.get(state.run_id, -1) + 1
        self._seq[state.run_id] = seq
        if type_ != "token":  # persist everything but per-token deltas (replay uses `done`)
            self.store.add_event(state.run_id, seq, type_, data)
        ev = {"run_id": state.run_id, "seq": seq, "ts": time.time(), "type": type_, "data": data}
        if q := self._queues.get(state.run_id):
            q.put_nowait(ev)
        return ev

    def checkpoint(self, state: RunState) -> None:
        self.store.checkpoint(state.run_id, state.session_id, state.model_dump_json(), state.status)

    def new_run(self, session_id: str, question: str) -> RunState:
        self.store.ensure_session(session_id)
        state = RunState(run_id=uuid.uuid4().hex[:12], session_id=session_id, question=question,
                         max_steps=settings.max_steps)
        self.checkpoint(state)
        return state

    def resume(self, run_id: str) -> RunState | None:
        raw = self.store.load_run(run_id)
        return RunState.model_validate_json(raw) if raw else None

    async def run(self, state: RunState) -> AsyncIterator[Event]:
        """Yield events live while the graph executes in a background task."""
        q: asyncio.Queue = asyncio.Queue()
        self._queues[state.run_id] = q
        task = asyncio.create_task(self._drive(state))
        task.add_done_callback(lambda _: q.put_nowait(None))  # sentinel
        try:
            while (ev := await q.get()) is not None:
                yield ev
            await task  # surface unexpected exceptions
        finally:
            self._queues.pop(state.run_id, None)
            if not task.done():  # client disconnected -> cancel the run
                task.cancel()
                state.status = "cancelled"
                self.checkpoint(state)

    async def _drive(self, state: RunState) -> None:
        """Execute nodes from state.node until done. Resumable: state.node is the resume point."""
        if state.node == "route":
            self.store.add_message(state.session_id, "user", state.question, state.run_id)
        while state.node != "done" and state.status == "running":
            current = state.node
            self.emit(state, "step", {"node": current, "phase": "start"})
            try:
                state.node = await self.nodes[current](state)
            except Exception as e:  # any node failure -> deterministic fallback (once)
                state.errors.append(f"{current}: {e}")
                self.emit(state, "error", {"node": current, "message": str(e)[:300]})
                if current == "fallback":
                    state.status = "failed"
                    break
                state.node = "fallback"
            self.checkpoint(state)
        if state.status == "running":
            state.status = "done"
            self.store.add_message(state.session_id, "assistant", state.answer, state.run_id)
        self.emit(state, "done", {"answer": state.answer, "citations": state.citations, "status": state.status})
        self.checkpoint(state)

    # ---------------- nodes ----------------
    async def route(self, state: RunState) -> str:
        msgs = self.ctx.build(instructions=ROUTER_INSTRUCTIONS, question=state.question,
                              turns=self.store.history(state.session_id, 6))
        decision = await self.llm.structured(msgs, RouteDecision, model=settings.router, max_tokens=200)
        state.route, state.search_query = decision.route, decision.search_query or state.question
        self.emit(state, "route", {"route": decision.route.value, "query": state.search_query,
                                   "reasoning": decision.reasoning})
        return "answer" if decision.route == Route.chat else "retrieve"

    async def retrieve(self, state: RunState) -> str:
        """Deterministic seed retrieval: cheap, reliable, and gives the loop a head start."""
        result = await self.tools.search_papers(state, state.search_query)
        self.emit(state, "tool_result", {"tool": "search_papers", "summary": result[:500]})
        for cand in state.candidates[:2 if state.route == Route.compare else 1]:
            self.tools.read_section(state, cand["node_id"])
        for ev in state.evidence.values():
            self.emit(state, "source", {"node_id": ev.node_id, "paper_id": ev.paper_id,
                                        "breadcrumb": ev.breadcrumb, "score": round(ev.score, 4)})
        return "answer" if state.route == Route.search else "research"

    async def research(self, state: RunState) -> str:
        """Bounded tool loop. The SLM chooses actions; code enforces caps and dedupe."""
        seen = {s["sig"] for s in state.scratchpad}
        while state.step < state.max_steps:
            state.step += 1
            progress = "\n".join(f"{s['tool']}({s['arg']}) -> {s['result'][:160]}" for s in state.scratchpad[-6:])
            cands = "\n".join(f"{c['node_id']} | {c['breadcrumb']}" for c in state.candidates[:6])
            msgs = self.ctx.build(instructions=RESEARCH_INSTRUCTIONS, question=state.question,
                                  evidence=list(state.evidence.values()),
                                  scratchpad=f"{progress}\ncandidates:\n{cands}".strip())
            call = await self.llm.structured(msgs, ToolCall, temperature=0.0, max_tokens=250)
            sig = state.call_signature(call)
            if call.tool == "finish":
                break
            if sig in seen:  # small models loop; a repeat means it has nothing better to do
                self.emit(state, "step", {"node": "research", "note": "repeated action; finishing"})
                break
            seen.add(sig)
            self.emit(state, "tool_call", {"tool": call.tool, "query": call.query, "node_id": call.node_id,
                                           "reasoning": call.reasoning})
            before = set(state.evidence)
            result = await self.tools.dispatch(state, call)
            state.scratchpad.append({"sig": sig, "tool": call.tool, "arg": call.query or call.node_id,
                                     "result": result})
            self.emit(state, "tool_result", {"tool": call.tool, "summary": result[:500]})
            for nid in set(state.evidence) - before:
                ev = state.evidence[nid]
                self.emit(state, "source", {"node_id": nid, "paper_id": ev.paper_id,
                                            "breadcrumb": ev.breadcrumb, "score": round(ev.score, 4)})
            self.checkpoint(state)
        return "answer"

    async def answer(self, state: RunState) -> str:
        evidence = list(state.evidence.values()) if state.route != Route.chat else None
        msgs = self.ctx.build(instructions=ANSWER_INSTRUCTIONS, question=state.question, evidence=evidence,
                              summary=self.store.session_summary(state.session_id),
                              turns=self.store.history(state.session_id, 10),
                              facts=self.store.top_facts(None), notes=self.store.notes(state.session_id))
        parts: list[str] = []
        async for delta in self.llm.stream(msgs, temperature=0.5, max_tokens=settings.budgets.output_reserve):
            parts.append(delta)
            self.emit(state, "token", {"delta": delta})
        state.answer = "".join(parts).strip()
        state.citations = [nid for nid in state.evidence if f"n:{nid}" in state.answer]
        return "verify" if state.evidence else "done"

    async def verify(self, state: RunState) -> str:
        msgs = self.ctx.build(instructions=VERIFY_INSTRUCTIONS, evidence=list(state.evidence.values()),
                              question=f"Draft answer to check:\n{state.answer}")
        try:
            result = await self.llm.structured(msgs, Verification, max_tokens=600)
        except LLMError as e:  # verification is best-effort; never block the answer on it
            self.emit(state, "verify", {"skipped": True, "reason": str(e)[:200]})
            return "done"
        unsupported = [c.claim for c in result.checks if not c.supported]
        if unsupported:
            state.answer += "\n\n_Unverified against the sources:_ " + "; ".join(unsupported)
        self.emit(state, "verify", {"checked": len(result.checks), "unsupported": unsupported})
        return "done"

    async def fallback(self, state: RunState) -> str:
        """Deterministic path that needs no structured output from the model."""
        if not state.evidence:
            hits = self.store.search(state.question, (await self.llm.embed([state.question]))[0], k=3)
            state.candidates = [h.__dict__ for h in hits]
            for h in hits[:2]:
                self.tools.read_section(state, h.node_id)
        state.route = state.route or Route.deep_read
        await self.answer(state)
        return "done"

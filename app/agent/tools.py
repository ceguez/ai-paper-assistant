"""The agent's tools. Few, non-overlapping, token-efficient, and they never raise into the
model: failures come back as short tool results the model can react to."""
from __future__ import annotations

from ..db import Store
from ..ingest.tree import count_tokens
from ..llm import LLM
from .state import Evidence, RunState, ToolCall


class Tools:
    def __init__(self, store: Store, llm: LLM, section_budget: int = 1800):
        self.store, self.llm, self.section_budget = store, llm, section_budget

    async def search_papers(self, state: RunState, query: str, k: int = 6) -> str:
        if not query.strip():
            return "error: empty query; provide keywords."
        qvec = (await self.llm.embed([query]))[0]
        hits = self.store.search(query, qvec, k=k)
        state.candidates = [h.__dict__ for h in hits]
        if not hits:
            return "no results; try different keywords."
        return "\n".join(f"{h.node_id} | {h.breadcrumb} | score={h.score:.3f}" for h in hits)

    def read_section(self, state: RunState, node_id: str) -> str:
        """Pointer -> whole section. Expands to the node's own text plus its subsections
        until the section budget is met (the Proxy-Pointer 'load the full section' step)."""
        row = self.store.node(node_id)
        if row is None:
            known = ", ".join(c["node_id"] for c in state.candidates[:5]) or "none"
            return f"error: unknown node_id '{node_id}'. Known candidates: {known}"
        parts, used = [row["text"]], count_tokens(row["text"] or "")
        for child in self.store.children(node_id):
            if child["kind"] != "section":
                continue
            cost = count_tokens(child["text"] or "")
            if used + cost > self.section_budget:
                parts.append(f"[more: subsection {child['node_id']} '{child['title']}']")
                continue
            parts.append(f"### {child['title']}\n{child['text']}")
            used += cost
        text = "\n\n".join(p for p in parts if p).strip()
        score = next((c["score"] for c in state.candidates if c["node_id"] == node_id), 0.0)
        state.evidence[node_id] = Evidence(node_id=node_id, paper_id=row["paper_id"],
                                           breadcrumb=row["breadcrumb"], text=text, score=score)
        return f"read {row['breadcrumb']} ({used} tokens) - added to sources."

    def get_element(self, state: RunState, node_id: str) -> str:
        row = self.store.node(node_id)
        if row is None or row["kind"] not in ("table", "figure"):
            return f"error: '{node_id}' is not a table or figure."
        state.evidence[node_id] = Evidence(node_id=node_id, paper_id=row["paper_id"],
                                           breadcrumb=row["breadcrumb"], text=row["text"])
        return f"added {row['kind']} '{row['title']}' to sources."

    def save_note(self, state: RunState, claim: str, node_id: str) -> str:
        if node_id not in state.evidence:
            return "error: notes must cite a section you have read (node_id in sources)."
        nid = self.store.add_note(state.session_id, claim, state.evidence[node_id].breadcrumb, node_id)
        return f"note {nid} saved."

    async def dispatch(self, state: RunState, call: ToolCall) -> str:
        match call.tool:
            case "search_papers":
                return await self.search_papers(state, call.query)
            case "read_section":
                return self.read_section(state, call.node_id)
            case "get_element":
                return self.get_element(state, call.node_id)
            case "save_note":
                return self.save_note(state, call.claim, call.node_id)
            case _:
                return "finish"

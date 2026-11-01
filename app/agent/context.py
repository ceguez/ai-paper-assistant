"""ContextAssembler: the one place that decides what enters the window.

Fills fixed slots in a fixed order (stable prefix -> KV-cache reuse) and enforces per-slot
token budgets. Overflow is trimmed from the lowest-value end of each slot, never by
letting the runtime truncate the prompt.
"""
from __future__ import annotations

from ..config import Budgets
from ..ingest.tree import count_tokens
from .state import Evidence

SYSTEM_BASE = (
    "<role>You are a meticulous research assistant answering questions about the user's paper library.</role>\n"
    "<rules>\n- Use only the sources provided; if they do not answer the question, say so.\n"
    "- Cite sources inline with their ids, like [n:ab12cd34].\n- Be concise and precise.\n</rules>"
)


def _trim(text: str, budget: int) -> str:
    if count_tokens(text) <= budget:
        return text
    words = text.split()
    keep = int(budget / 1.33)
    return " ".join(words[:keep]) + " …[truncated]"


class ContextAssembler:
    def __init__(self, budgets: Budgets):
        self.b = budgets

    def sources_block(self, evidence: list[Evidence]) -> str:
        """Highest-scoring sections first; drop whole sections before truncating one."""
        out, used = [], 0
        for ev in sorted(evidence, key=lambda e: -e.score):
            page = f' page="{ev.page}"' if ev.page else ""
            block = f'<source id="n:{ev.node_id}" path="{ev.breadcrumb}"{page}>\n{ev.text}\n</source>'
            cost = count_tokens(block)
            if used + cost > self.b.sources:
                remaining = self.b.sources - used
                if remaining > 200 and not out:  # a single oversized section: send its head
                    out.append(_trim(block, remaining))
                break
            out.append(block)
            used += cost
        return "<sources>\n" + "\n".join(out) + "\n</sources>" if out else "<sources>none</sources>"

    def history_block(self, summary: str, turns: list[dict]) -> str:
        """Rolling summary + most recent turns that fit; oldest turns drop first."""
        kept, used = [], count_tokens(summary)
        for t in reversed(turns):
            line = f"{t['role']}: {t['content']}"
            cost = count_tokens(line)
            if used + cost > self.b.history:
                break
            kept.append(line)
            used += cost
        parts = ([f"<summary>{summary}</summary>"] if summary else []) + list(reversed(kept))
        return "<history>\n" + "\n".join(parts) + "\n</history>" if parts else ""

    def memory_block(self, facts: list[str], notes: list[dict]) -> str:
        lines = [f"- {f}" for f in facts] + [f"- note: {n['claim']} [n:{n['node_id']}]" for n in notes]
        return f"<memory>\n{_trim(chr(10).join(lines), self.b.memory)}\n</memory>" if lines else ""

    def build(self, *, instructions: str, question: str, evidence: list[Evidence] | None = None,
              summary: str = "", turns: list[dict] | None = None, facts: list[str] | None = None,
              notes: list[dict] | None = None, scratchpad: str = "") -> list[dict]:
        system = _trim(f"{SYSTEM_BASE}\n<task>{instructions}</task>", self.b.system + self.b.tools)
        user_parts = [
            self.memory_block(facts or [], notes or []),
            self.history_block(summary, turns or []),
            self.sources_block(evidence or []) if evidence is not None else "",
            f"<progress>\n{_trim(scratchpad, 400)}\n</progress>" if scratchpad else "",
            f"<question>{question}</question>",  # question last: small models attend to the end
        ]
        return [{"role": "system", "content": system},
                {"role": "user", "content": "\n\n".join(p for p in user_parts if p)}]

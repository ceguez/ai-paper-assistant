"""Typed run state and every schema the SLM is constrained to.

Keep schemas flat, short and enum-heavy: small models are far more reliable choosing from
an enum than writing free-form structure. `reasoning` comes first so the model 'thinks'
briefly before committing to the decision field.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Route(str, Enum):
    chat = "chat"            # small talk / meta questions, no retrieval
    search = "search"        # find which papers/sections discuss X
    deep_read = "deep_read"  # answer a question that needs reading sections closely
    compare = "compare"      # contrast two or more papers


class RouteDecision(BaseModel):
    reasoning: str = Field(max_length=300)
    route: Route
    search_query: str = Field(default="", max_length=200, description="Keyword-rich query for retrieval")


class ToolCall(BaseModel):
    """One step of the research loop. A discriminated choice keeps the action space tiny."""
    reasoning: str = Field(max_length=300)
    tool: Literal["search_papers", "read_section", "get_element", "save_note", "finish"]
    query: str = Field(default="", max_length=200)
    node_id: str = Field(default="", max_length=40)
    claim: str = Field(default="", max_length=400)


class ClaimCheck(BaseModel):
    claim: str
    supported: bool
    node_ids: list[str] = Field(default_factory=list)


class Verification(BaseModel):
    checks: list[ClaimCheck] = Field(default_factory=list, max_length=12)


class Evidence(BaseModel):
    node_id: str
    paper_id: str
    breadcrumb: str
    text: str
    score: float = 0.0
    page: int | None = None


class RunState(BaseModel):
    run_id: str
    session_id: str
    question: str
    paper_ids: list[str] = Field(default_factory=list)   # empty = whole library
    route: Route | None = None
    search_query: str = ""
    step: int = 0
    max_steps: int = 6
    node: str = "route"                                 # next node to execute (resume point)
    scratchpad: list[dict] = Field(default_factory=list)  # tool calls + short results this run
    evidence: dict[str, Evidence] = Field(default_factory=dict)
    candidates: list[dict] = Field(default_factory=list)  # search hits not yet read
    answer: str = ""
    citations: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    status: Literal["running", "done", "failed", "cancelled"] = "running"

    def call_signature(self, call: ToolCall) -> str:
        return f"{call.tool}|{call.query.strip().lower()}|{call.node_id}"

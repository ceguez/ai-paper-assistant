import asyncio

from app.agent.context import ContextAssembler
from app.agent.graph import Orchestrator
from app.config import Budgets
from app.db import Store
from app.ingest.pipeline import ingest_markdown
from app.ingest.tree import build_tree, pointer_text, size_nodes, walk
from app.llm import LLMError

from .fake_llm import FakeLLM

PAPER = """# Sparse Mixture Routers for Small Models

## Abstract
We study expert routing in small language models and show load balancing matters.

## 1 Introduction
Mixture of experts scales capacity without scaling compute per token.

## 2 Method
### 2.1 Router
The router is a linear layer with top-2 gating and an auxiliary load balancing loss.

Table 1: Hyperparameters
| param | value |
| --- | --- |
| experts | 8 |
| top-k | 2 |

### 2.2 Training
We train for 100k steps with AdamW and cosine decay. The learning rate is 3e-4.

Figure 2: Expert utilization over training.

## 3 Results
Load balancing improves perplexity by 4 percent and stabilises expert utilisation.

## Acknowledgements
We thank our funders.
"""

META = {"title": "Sparse Mixture Routers for Small Models", "authors": ["A. Author"], "year": 2026,
        "doi": "not-a-doi", "arxiv_id": "2601.01234"}


def run(coro):
    return asyncio.run(coro)


def test_skeleton_tree_breadcrumbs_and_elements():
    root = build_tree(PAPER, "p1")
    size_nodes(root)
    nodes = list(walk(root))
    crumbs = {n.breadcrumb for n in nodes}
    assert root.title == "Sparse Mixture Routers for Small Models"
    assert "Sparse Mixture Routers for Small Models > 2 Method > 2.1 Router" in crumbs
    kinds = {n.kind for n in nodes}
    assert {"table", "figure"} <= kinds
    table = next(n for n in nodes if n.kind == "table")
    assert "Table 1" in table.title and "| experts | 8 |" in table.text
    ack = next(n for n in nodes if n.title == "Acknowledgements")
    assert ack.embeddable is False
    assert pointer_text(table).startswith("[table]")


def test_oversized_section_is_split():
    big = "# T\n\n## Long\n" + "\n\n".join("word " * 300 for _ in range(8))
    root = build_tree(big, "p2")
    size_nodes(root, max_tokens=500)
    long = next(n for n in walk(root) if n.title == "Long")
    assert len(long.children) >= 3 and all("(part" in c.title for c in long.children)


def test_ingest_and_hybrid_search_finds_the_right_section():
    store, llm = Store(":memory:"), FakeLLM({"PaperMetadata": [META]})
    n = run(ingest_markdown(store, llm, PAPER, "p1"))
    assert n > 5
    paper = store.conn.execute("SELECT * FROM papers").fetchone()
    assert paper["doi"] is None and paper["arxiv_id"] == "2601.01234"  # validator rejected the bad DOI
    hits = store.search("learning rate AdamW", run(llm.embed(["learning rate AdamW"]))[0], k=3)
    assert hits and hits[0].breadcrumb.endswith("2.2 Training")


def test_context_budget_drops_whole_sections_first():
    from app.agent.state import Evidence
    ctx = ContextAssembler(Budgets(num_ctx=4096, sources=300))
    ev = [Evidence(node_id=f"n{i}", paper_id="p", breadcrumb=f"S{i}", text="tok " * 150, score=1 - i / 10)
          for i in range(3)]
    block = ctx.sources_block(ev)
    assert 'id="n:n0"' in block and 'id="n:n2"' not in block


def _orchestrator(script, answer="Training uses AdamW at 3e-4 [n:{nid}]."):
    store = Store(":memory:")
    llm = FakeLLM({"PaperMetadata": [META]})
    run(ingest_markdown(store, llm, PAPER, "p1"))
    training = store.conn.execute("SELECT node_id FROM nodes WHERE title='2.2 Training'").fetchone()[0]
    llm.script.update({k: __import__("collections").deque(v(training) if callable(v) else v)
                       for k, v in script.items()})
    llm.answer = answer.format(nid=training)
    return store, llm, Orchestrator(store, llm), training


async def _collect(agent, state):
    return [ev async for ev in agent.run(state)]


def test_full_deep_read_run_streams_events_and_cites():
    store, llm, agent, training = _orchestrator({
        "RouteDecision": [{"reasoning": "needs paper content", "route": "deep_read",
                           "search_query": "training optimizer learning rate"}],
        "ToolCall": lambda nid: [
            {"reasoning": "read it", "tool": "read_section", "node_id": nid},
            {"reasoning": "record", "tool": "save_note", "claim": "lr is 3e-4", "node_id": nid},
            {"reasoning": "enough", "tool": "finish"}],
        "Verification": [{"checks": [{"claim": "AdamW at 3e-4", "supported": True, "node_ids": []}]}],
    })
    state = agent.new_run("s1", "What optimizer and learning rate are used?")
    events = run(_collect(agent, state))
    types = [e["type"] for e in events]
    assert types[0] == "step" and types[-1] == "done"
    assert {"route", "tool_call", "source", "token", "verify"} <= set(types)
    done = events[-1]["data"]
    assert done["status"] == "done" and training in done["citations"]
    assert store.notes("s1")[0]["claim"] == "lr is 3e-4"
    assert [e["seq"] for e in events] == sorted(e["seq"] for e in events)  # ordered stream
    resumed = agent.resume(state.run_id)
    assert resumed.status == "done" and resumed.node == "done"  # checkpointed


def test_loop_detection_stops_repeated_tool_calls():
    _, llm, agent, _ = _orchestrator({
        "RouteDecision": [{"reasoning": "x", "route": "deep_read", "search_query": "router gating"}],
        "ToolCall": [{"reasoning": "a", "tool": "search_papers", "query": "router"}] * 5,
    })
    run(_collect(agent, agent.new_run("s2", "How does the router work?")))
    assert llm.calls.count("ToolCall") == 2  # second identical call triggers finish


def test_schema_failure_falls_back_to_grounded_answer():
    store, llm, agent, training = _orchestrator({
        "RouteDecision": [LLMError("schema RouteDecision invalid after retries")],
    })
    events = run(_collect(agent, agent.new_run("s3", "What learning rate is used in training?")))
    types = [e["type"] for e in events]
    assert "error" in types and events[-1]["data"]["status"] == "done"
    assert any(e["type"] == "step" and e["data"]["node"] == "fallback" for e in events)
    assert "3e-4" in events[-1]["data"]["answer"]


def test_chat_route_skips_retrieval():
    _, llm, agent, _ = _orchestrator({"RouteDecision": [{"reasoning": "greeting", "route": "chat"}]},
                                     answer="Hello! Ask me about your papers.")
    events = run(_collect(agent, agent.new_run("s4", "hi there")))
    assert not any(e["type"] == "source" for e in events)
    assert "ToolCall" not in llm.calls and "Verification" not in llm.calls

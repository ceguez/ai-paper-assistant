import asyncio
from collections import deque

from app.agent.context import ContextAssembler
from app.agent.graph import Orchestrator
from app.agent.state import Evidence
from app.config import Budgets
from app.db import Store
from app.ingest.pipeline import ingest_markdown
from app.ingest.tree import build_tree, pointer_text, size_nodes, walk
from app.llm import LLMError

from .conftest import META, META_B, PAPER, PAPER_B
from .fake_llm import FakeLLM

TITLE = "Sparse Mixture Routers for Small Models"


def run(coro):
    return asyncio.run(coro)


# ---------------- skeleton tree ----------------
def test_skeleton_tree_breadcrumbs_pages_and_elements():
    root = build_tree(PAPER, "p1")
    size_nodes(root)
    nodes = {n.title: n for n in walk(root)}
    assert root.title == TITLE
    training = nodes["2.2 Training"]
    assert training.breadcrumb == f"{TITLE} > 2 Method > 2.2 Training"
    assert (training.page_start, training.page_end) == (3, 3)
    assert nodes["2 Method"].page_start == 2
    table = next(n for n in walk(root) if n.kind == "table")
    assert "Table 1" in table.title and "| experts | 8 |" in table.text and table.page_start == 2
    assert any(n.kind == "figure" for n in walk(root))
    assert nodes["Acknowledgements"].embeddable is False
    assert nodes["Front matter"].page_start == 1 and "Ada Author" in nodes["Front matter"].text
    assert pointer_text(table).startswith("[table]")


def test_bold_numbered_lines_become_sections():
    md = "<!-- page:1 -->\n# A Paper\n\n**1 Introduction**\nintro text here\n\n**2.1 Setup**\nsetup text here"
    nodes = {n.title: n for n in walk(build_tree(md, "p"))}
    assert nodes["2.1 Setup"].level == 3 and nodes["1 Introduction"].level == 2


def test_document_without_headings_falls_back_to_pages():
    md = "<!-- page:1 -->\nplain text one\n<!-- page:2 -->\nplain text two"
    nodes = {n.title: n for n in walk(build_tree(md, "p"))}
    assert nodes["Page 2"].page_start == 2 and "two" in nodes["Page 2"].text


def test_oversized_section_is_split_with_pages():
    big = "<!-- page:5 -->\n# T\n\n## Long\n" + "\n\n".join("word " * 300 for _ in range(8))
    root = build_tree(big, "p2")
    size_nodes(root, max_tokens=500)
    long = next(n for n in walk(root) if n.title == "Long")
    assert len(long.children) >= 3 and all("(part" in c.title and c.page_start == 5 for c in long.children)


# ---------------- store ----------------
def _library():
    store = Store(":memory:")
    llm = FakeLLM({"PaperMetadata": [META, META_B]})
    run(ingest_markdown(store, llm, PAPER, "p1", filename="routers.pdf"))
    run(ingest_markdown(store, llm, PAPER_B, "p2", filename="heads.pdf"))
    return store, llm


def test_ingest_and_hybrid_search_finds_the_right_section():
    store, llm = _library()
    paper = store.paper("p1")
    assert paper["status"] == "ready" and paper["doi"] is None and paper["arxiv_id"] == "2601.01234"
    q = "learning rate AdamW"
    hits = store.search(q, run(llm.embed([q]))[0], k=3)
    assert hits and hits[0].breadcrumb.endswith("2.2 Training")


def test_natural_language_question_ranks_the_right_section_first():
    store, llm = _library()
    q = "What optimizer and learning rate are used in training?"
    assert store.search(q, run(llm.embed([q]))[0], k=3)[0].breadcrumb.endswith("2.2 Training")


def test_search_scoped_to_selected_papers():
    store, llm = _library()
    q = "optimizer learning rate"
    scoped = store.search(q, run(llm.embed([q]))[0], k=5, paper_ids=["p2"])
    assert scoped and all(h.paper_id == "p2" for h in scoped)


def test_delete_paper_removes_nodes_pointers_and_fts_rows():
    store, llm = _library()
    store.delete_paper("p1")
    assert store.q1("SELECT COUNT(*) AS n FROM nodes WHERE paper_id='p1'")["n"] == 0
    assert store.q1("SELECT COUNT(*) AS n FROM pointers WHERE paper_id='p1'")["n"] == 0
    fts = store.q("SELECT rowid FROM pointers_fts WHERE pointers_fts MATCH 'AdamW'")
    assert fts == []  # external-content FTS kept in sync by trigger on cascade delete
    q = "AdamW learning rate"
    assert all(h.paper_id == "p2" for h in store.search(q, run(llm.embed([q]))[0]))


def test_reingest_replaces_old_index():
    store, llm = _library()
    before = store.q1("SELECT COUNT(*) AS n FROM pointers WHERE paper_id='p1'")["n"]
    llm.script["PaperMetadata"] = deque([META])
    run(ingest_markdown(store, llm, PAPER, "p1"))
    assert store.q1("SELECT COUNT(*) AS n FROM pointers WHERE paper_id='p1'")["n"] == before


def test_context_budget_drops_whole_sections_first():
    ctx = ContextAssembler(Budgets(num_ctx=4096, sources=300))
    ev = [Evidence(node_id=f"n{i}", paper_id="p", breadcrumb=f"S{i}", text="tok " * 150, score=1 - i / 10, page=i)
          for i in range(3)]
    block = ctx.sources_block(ev)
    assert 'id="n:n0"' in block and 'page="' not in block.split('id="n:n0"')[0] and 'id="n:n2"' not in block


# ---------------- orchestration ----------------
def _orchestrator(script, answer="Training uses AdamW at 3e-4 [n:{nid}]."):
    store, llm = _library()
    training = store.q1("SELECT node_id FROM nodes WHERE title='2.2 Training'")[0]
    for k, v in script.items():
        llm.script[k] = deque(v(training) if callable(v) else v)
    llm.answer = answer.format(nid=training)
    return store, llm, Orchestrator(store, llm, llm.settings), training


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
    src = next(e["data"] for e in events if e["type"] == "source" and e["data"]["node_id"] == training)
    assert src["page"] == 3 and src["paper_title"] == TITLE
    done = events[-1]["data"]
    assert done["status"] == "done" and training in done["citations"]
    assert store.notes("s1")[0]["claim"] == "lr is 3e-4"
    assert [e["seq"] for e in events] == list(range(len(events)))
    persisted = store.events_since(state.run_id)
    assert persisted[-1]["type"] == "done" and not any(e["type"] == "token" for e in persisted)
    assert [m["role"] for m in store.messages("s1")] == ["user", "assistant"]


def test_run_scoped_to_one_paper_never_reads_the_other():
    _, llm, agent, _ = _orchestrator({
        "RouteDecision": [{"reasoning": "x", "route": "search", "search_query": "optimizer learning rate"}]})
    state = agent.new_run("s5", "Which optimizer?", paper_ids=["p2"])
    events = run(_collect(agent, state))
    sources = [e["data"] for e in events if e["type"] == "source"]
    assert sources and all(s["paper_id"] == "p2" for s in sources)


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


def test_resubscribe_replays_from_sequence_and_cancel_keeps_partial_answer():
    _, llm, agent, _ = _orchestrator({"RouteDecision": [{"reasoning": "g", "route": "chat"}]},
                                     answer=" ".join(f"w{i}" for i in range(200)))
    llm.token_delay = 0.005

    async def scenario():
        state = agent.new_run("s6", "hello")
        agent.start(state)
        first = []
        async for ev in agent.events(state.run_id):
            first.append(ev)
            if len(first) == 5:
                break  # client went away (browser refresh): the run keeps going
        assert agent.busy("s6") == state.run_id
        rest = []
        async for ev in agent.events(state.run_id, after_seq=first[-1]["seq"]):
            rest.append(ev)
            if sum(e["type"] == "token" for e in rest) == 10:
                agent.cancel(state.run_id)
        assert rest[0]["seq"] == first[-1]["seq"] + 1
        done = rest[-1]["data"]
        assert rest[-1]["type"] == "done" and done["status"] == "cancelled"
        assert agent.busy("s6") is None
        return done

    run(scenario())


def test_restart_closes_interrupted_runs():
    store, llm, agent, _ = _orchestrator({})
    state = agent.new_run("s7", "q")  # checkpointed as running, never started
    assert agent.recover_interrupted() == 1
    assert store.events_since(state.run_id)[-1]["data"]["status"] == "failed"

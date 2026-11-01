"""End-to-end through the HTTP API: upload -> background ingestion -> streamed run -> replay
-> delete. PDF parsing is patched (no PyMuPDF here); everything after it is real."""
import json
import time

import pytest
from starlette.testclient import TestClient

from app.agent.graph import Orchestrator
from app.api import Services, create_app
from app.db import Store
from app.ingest import pipeline
from app.ingest.parser import ParsedDoc
from app.jobs import IngestWorker

from .conftest import META, META_B, PAPER, PAPER_B
from .fake_llm import FakeLLM

PDF_A = b"%PDF-1.4 paper A"
PDF_B = b"%PDF-1.4 paper B"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    docs = {PDF_A: PAPER, PDF_B: PAPER_B}

    def fake_parse(path, preference="auto"):
        data = open(path, "rb").read()
        if data not in docs:
            raise RuntimeError("could not parse PDF: corrupt")
        return ParsedDoc(docs[data], "fake", 4)

    monkeypatch.setattr(pipeline, "parse_pdf", fake_parse)
    llm = FakeLLM({"PaperMetadata": [META, META_B]}, answer="Uses AdamW [n:{nid}].")
    from dataclasses import replace
    llm.settings = replace(llm.settings, data_dir=str(tmp_path))
    store = Store(str(tmp_path / "t.db"))
    svc = Services(llm.settings, store, llm, Orchestrator(store, llm, llm.settings), IngestWorker(store, llm))
    with TestClient(create_app(svc)) as c:
        c.llm = llm
        yield c


def wait_ready(client, n, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        papers = client.get("/api/papers").json()
        if len(papers) >= n and all(p["status"] in ("ready", "failed") for p in papers):
            return papers
        time.sleep(0.05)
    raise AssertionError(f"papers not ready: {papers}")


def sse_events(resp):
    events = []
    for line in resp.iter_lines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


def test_health_reports_models_and_memory_fit(client):
    h = client.get("/api/health").json()
    assert h["ok"] and h["gen_present"] and h["embed_present"]
    assert h["num_ctx"] == 16384 and 0 < h["memory_estimate_gb"] <= h["gpu_budget_gb"]


def test_upload_validates_and_dedupes(client):
    r = client.post("/api/papers", files=[("files", ("a.pdf", PDF_A, "application/pdf")),
                                          ("files", ("notes.txt", b"hello", "text/plain"))])
    assert r.status_code == 202
    results = r.json()["papers"]
    assert results[0]["status"] == "queued" and results[1]["error"] == "not a PDF"
    wait_ready(client, 1)
    again = client.post("/api/papers", files=[("files", ("copy.pdf", PDF_A, "application/pdf"))]).json()
    assert again["papers"][0]["status"] == "ready"  # same bytes -> same paper, no re-ingest
    assert len(client.get("/api/papers").json()) == 1


def test_failed_parse_is_reported_and_retryable(client):
    client.post("/api/papers", files=[("files", ("bad.pdf", b"%PDF-1.4 corrupt", "application/pdf"))])
    [paper] = wait_ready(client, 1)
    assert paper["status"] == "failed" and "corrupt" in paper["error"]
    assert client.post(f"/api/papers/{paper['paper_id']}/retry").json()["status"] == "queued"


def test_full_run_over_http_with_scope_replay_and_delete(client):
    client.post("/api/papers", files=[("files", ("a.pdf", PDF_A, "application/pdf")),
                                      ("files", ("b.pdf", PDF_B, "application/pdf"))])
    papers = {p["title"]: p for p in wait_ready(client, 2)}
    a = papers["Sparse Mixture Routers for Small Models"]
    detail = client.get(f"/api/papers/{a['paper_id']}").json()
    training = next(n for n in detail["tree"] if n["title"] == "2.2 Training")
    assert training["page_start"] == 3
    client.llm.answer = f"Training uses AdamW at 3e-4 [n:{training['node_id']}]."
    client.llm.script["RouteDecision"].append({"reasoning": "x", "route": "deep_read",
                                               "search_query": "training optimizer learning rate"})
    client.llm.script["ToolCall"].append({"reasoning": "done", "tool": "finish"})

    sid = client.post("/api/sessions").json()["session_id"]
    r = client.post(f"/api/sessions/{sid}/runs", json={"message": "Which optimizer?", "paper_ids": [a["paper_id"]]})
    assert r.status_code == 202
    run_id = r.json()["run_id"]
    with client.stream("GET", f"/api/runs/{run_id}/events") as resp:
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = sse_events(resp)
    assert events[-1]["type"] == "done" and events[-1]["data"]["status"] == "done"
    sources = [e["data"] for e in events if e["type"] == "source"]
    assert sources and all(s["paper_id"] == a["paper_id"] for s in sources)
    assert training["node_id"] in events[-1]["data"]["citations"]

    # Reconnect with Last-Event-ID: only later events come back.
    with client.stream("GET", f"/api/runs/{run_id}/events", headers={"Last-Event-ID": "3"}) as resp:
        replay = sse_events(resp)
    assert replay[0]["seq"] == 4 and replay[-1]["type"] == "done"

    msgs = client.get(f"/api/sessions/{sid}/messages").json()
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert client.get("/api/sessions").json()[0]["title"] == "Which optimizer?"

    pdf = client.get(f"/api/papers/{a['paper_id']}/file")
    assert pdf.status_code == 200 and pdf.content == PDF_A
    assert client.delete(f"/api/papers/{a['paper_id']}").status_code == 200
    assert client.get(f"/api/papers/{a['paper_id']}").status_code == 404


def test_bad_requests(client):
    sid = client.post("/api/sessions").json()["session_id"]
    assert client.post(f"/api/sessions/{sid}/runs", json={"message": "  "}).status_code == 400
    assert client.get("/api/runs/nope/events").status_code == 404
    assert client.delete("/api/papers/nope").status_code == 404
    assert client.get("/").status_code == 200

"""Regression tests for the local-safety fixes: budget headroom, non-blocking parsing,
and runs that finish after the client disconnects."""
import asyncio
import time

import pytest

from app.agent.graph import Orchestrator
from app.config import Budgets, settings
from app.db import Store
from app.ingest import pipeline

from .fake_llm import FakeLLM


def test_budgets_leave_ten_percent_headroom():
    settings.budgets.check()
    assert settings.budgets.check is not None
    with pytest.raises(ValueError):
        Budgets(sources=3800).check()  # the old layout (8,100 of 8,192) is now rejected


def test_pdf_parsing_does_not_block_the_event_loop(monkeypatch, tmp_path):
    def slow_parse(path):
        time.sleep(0.5)  # stands in for Docling / PyMuPDF work
        raise RuntimeError("stop after parsing")

    monkeypatch.setattr(pipeline, "parse_pdf", slow_parse)
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")

    async def scenario():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.05)
                ticks += 1

        t = asyncio.create_task(ticker())
        with pytest.raises(RuntimeError):
            await pipeline.ingest_pdf(Store(":memory:"), FakeLLM(), str(pdf))
        t.cancel()
        return ticks

    assert asyncio.run(scenario()) >= 5  # the loop kept running during the 0.5 s parse


def test_run_finishes_and_hooks_fire_after_client_disconnects():
    done_hook = []
    llm = FakeLLM({"RouteDecision": [{"reasoning": "greeting", "route": "chat"}]},
                  answer=" ".join(f"w{i}" for i in range(50)))
    store = Store(":memory:")

    async def on_done(state):
        done_hook.append(state.run_id)

    agent = Orchestrator(store, llm, on_done=on_done)

    async def scenario():
        state = agent.new_run("s1", "hello")
        gen = agent.run(state)
        await gen.__anext__()          # client receives one event...
        await gen.aclose()             # ...then the tab closes
        for _ in range(100):           # the run keeps going in the background
            if store.load_run(state.run_id) and '"status":"done"' in store.load_run(state.run_id):
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)      # let the post-run hook task run
        return state

    state = asyncio.run(scenario())
    assert '"status":"done"' in store.load_run(state.run_id)
    assert [m["role"] for m in store.history("s1")] == ["user", "assistant"]
    assert store.events_since(state.run_id)[-1]["type"] == "done"   # a reconnect can replay it
    assert done_hook == [state.run_id]

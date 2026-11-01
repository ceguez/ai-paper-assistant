"""The real Ollama and OpenAI-compatible clients against mock servers that speak the same
wire format (httpx.MockTransport), so request shapes and parsing are tested without a GPU."""
import asyncio
import json
from dataclasses import replace

import httpx
import pytest
from pydantic import BaseModel

from app.config import Settings
from app.llm import LLMError, OllamaLLM, OpenAICompatLLM


class Pick(BaseModel):
    route: str
    n: int


def run(coro):
    return asyncio.run(coro)


def settings(**kw) -> Settings:
    return replace(Settings().resolved(["qwen2.5:7b-instruct"]), **kw)


def ollama_mock(log: list, chat_replies: list[str]):
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content) if req.content else {}
        log.append((req.url.path, body))
        if req.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen2.5:7b-instruct"},
                                                        {"name": "nomic-embed-text:latest"}]})
        if req.url.path == "/api/embed":
            return httpx.Response(200, json={"model": body["model"], "embeddings": [[0.1, 0.2]] * len(body["input"])})
        if req.url.path == "/api/chat" and body.get("stream"):
            lines = [{"message": {"role": "assistant", "content": t}, "done": False} for t in ("Hel", "lo")]
            lines.append({"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"})
            return httpx.Response(200, content="\n".join(json.dumps(x) for x in lines).encode())
        if req.url.path == "/api/chat":
            if body["model"] == "missing":
                return httpx.Response(404, json={"error": "model 'missing' not found"})
            return httpx.Response(200, json={"message": {"role": "assistant", "content": chat_replies.pop(0)},
                                             "done": True, "done_reason": "stop"})
        return httpx.Response(404)
    return httpx.MockTransport(handler)


def test_ollama_structured_sends_schema_and_ctx_then_reasks_on_invalid():
    log = []
    llm = OllamaLLM(settings(), transport=ollama_mock(log, ['{"route": "x"}', '{"route": "chat", "n": 2}']))
    out = run(llm.structured([{"role": "user", "content": "hi"}], Pick))
    assert out == Pick(route="chat", n=2)
    first, second = log[0][1], log[1][1]
    assert first["format"] == Pick.model_json_schema() and first["stream"] is False
    assert first["options"]["num_ctx"] == 16384 and first["options"]["temperature"] == 0.0
    assert "failed validation" in second["messages"][-1]["content"]  # re-ask carries the error


def test_ollama_strips_code_fences_and_disables_qwen3_thinking():
    log = []
    llm = OllamaLLM(settings(), transport=ollama_mock(log, ['```json\n{"route": "a", "n": 1}\n```']))
    assert run(llm.structured([{"role": "user", "content": "x"}], Pick, model="qwen3:8b")).n == 1
    assert log[0][1]["think"] is False


def test_ollama_stream_embed_and_tags():
    log = []
    llm = OllamaLLM(settings(), transport=ollama_mock(log, []))

    async def go():
        tokens = [t async for t in llm.stream([{"role": "user", "content": "x"}])]
        return tokens, await llm.embed(["a", "b"]), await llm.list_models()

    tokens, vecs, models = run(go())
    assert "".join(tokens) == "Hello" and len(vecs) == 2 and "qwen2.5:7b-instruct" in models
    assert log[1][1]["model"] == "nomic-embed-text"


def test_ollama_4xx_is_not_retried():
    log = []
    llm = OllamaLLM(settings(), transport=ollama_mock(log, []))
    with pytest.raises(LLMError, match="not found"):
        run(llm.structured([{"role": "user", "content": "x"}], Pick, model="missing"))
    assert len(log) == 1


def test_openai_compatible_server_json_schema_and_sse_stream():
    log = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content) if req.content else {}
        log.append(body)
        if req.url.path == "/v1/chat/completions" and body.get("stream"):
            chunks = [{"choices": [{"delta": {"content": t}}]} for t in ("Hi", " there")]
            sse = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, content=sse.encode())
        if req.url.path == "/v1/chat/completions":
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"route": "x", "n": 3}'}}]})
        if req.url.path == "/v1/embeddings":
            return httpx.Response(200, json={"data": [{"index": 1, "embedding": [2.0]}, {"index": 0, "embedding": [1.0]}]})
        return httpx.Response(200, json={"data": [{"id": "qwen2.5-7b-instruct"}]})

    llm = OpenAICompatLLM(settings(runtime="openai"), transport=httpx.MockTransport(handler))

    async def go():
        pick = await llm.structured([{"role": "user", "content": "x"}], Pick)
        tokens = [t async for t in llm.stream([{"role": "user", "content": "x"}])]
        return pick, "".join(tokens), await llm.embed(["a", "b"])

    pick, text, vecs = run(go())
    assert pick.n == 3 and text == "Hi there" and vecs == [[1.0], [2.0]]
    assert log[0]["response_format"]["json_schema"]["schema"] == Pick.model_json_schema()

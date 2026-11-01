"""A stand-in for Ollama that speaks its HTTP API (tags, embed, chat, streaming chat).

Use it to work on the UI or run the integration test without loading a model:
    python scripts/mock_ollama.py --port 11435
    PA_BASE_URL=http://127.0.0.1:11435 python -m app serve

Answers are assembled from the first source in the prompt, so citations, pages and the
research trail all behave like the real thing; the words are not a model's.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re

import numpy as np
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

MODELS = ["qwen2.5:7b-instruct", "nomic-embed-text:latest"]


def embed_text(text: str, dim: int = 256) -> list[float]:
    v = np.zeros(dim, dtype=np.float32)
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim] += 1.0
    n = np.linalg.norm(v)
    return (v / n if n else v).tolist()


def structured_reply(schema: dict, messages: list[dict]) -> dict:
    title = schema.get("title", "")
    user = messages[-1]["content"]
    question = re.search(r"<question>(.*?)</question>", user, re.S)
    q = question.group(1).strip() if question else user[:200]
    if title == "PaperMetadata":
        h1 = re.search(r"^#\s+(.+)$", user, re.M)
        return {"title": h1.group(1).strip() if h1 else "Untitled", "authors": ["Mock Author"], "year": 2026}
    if title == "RouteDecision":
        if re.match(r"^(hi|hello|thanks|thank you)\b", q, re.I):
            return {"reasoning": "A greeting.", "route": "chat", "search_query": ""}
        return {"reasoning": "Needs the content of the papers.", "route": "deep_read", "search_query": q}
    if title == "ToolCall":
        cands = re.findall(r"^([0-9a-f]{16}) \|", user, re.M)
        read = "-> read" in user
        if cands and not read:
            return {"reasoning": "This section looks most relevant.", "tool": "read_section", "node_id": cands[0]}
        return {"reasoning": "The sources answer the question.", "tool": "finish"}
    if title == "Verification":
        return {"checks": [{"claim": "The answer restates the cited section.", "supported": True, "node_ids": []}]}
    return {}


def answer_text(messages: list[dict]) -> str:
    user = messages[-1]["content"]
    src = re.search(r'<source id="n:([0-9a-f]+)" path="([^"]+)"[^>]*>\n(.*?)\n</source>', user, re.S)
    if not src:
        return "Hello. Add papers to the library and ask about them; answers will cite the sections they use."
    node, path, text = src.groups()
    sentence = re.split(r"(?<=[.!?])\s", " ".join(text.split()))[0]
    section = path.split(" > ")[-1]
    return (f"According to **{section}**, {sentence[0].lower() + sentence[1:]} [n:{node}]\n\n"
            f"- This comes straight from the section text, so check the page for context. [n:{node}]")


async def tags(request: Request):
    return JSONResponse({"models": [{"name": m} for m in MODELS]})


async def embed(request: Request):
    body = await request.json()
    inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
    return JSONResponse({"model": body["model"], "embeddings": [embed_text(t) for t in inputs]})


async def chat(request: Request):
    body = await request.json()
    msgs = body["messages"]
    if isinstance(body.get("format"), dict):
        return JSONResponse({"message": {"role": "assistant", "content": json.dumps(structured_reply(body["format"], msgs))},
                             "done": True, "done_reason": "stop"})
    text = answer_text(msgs)

    async def gen():
        for word in re.findall(r"\S+\s*", text):
            await asyncio.sleep(0.03)
            yield json.dumps({"message": {"role": "assistant", "content": word}, "done": False}) + "\n"
        yield json.dumps({"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"}) + "\n"

    if body.get("stream", True):
        return StreamingResponse(gen(), media_type="application/x-ndjson")
    return JSONResponse({"message": {"role": "assistant", "content": text}, "done": True})


app = Starlette(routes=[Route("/api/tags", tags), Route("/api/embed", embed, methods=["POST"]),
                        Route("/api/chat", chat, methods=["POST"])])

if __name__ == "__main__":
    import uvicorn

    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=11435)
    uvicorn.run(app, host="127.0.0.1", port=p.parse_args().port, log_level="warning")

"""Deterministic stand-in for a local model: scripted structured outputs per schema,
word-streamed answers, and hashed bag-of-words embeddings (so retrieval actually works)."""
from __future__ import annotations

import asyncio
import hashlib
import re
from collections import defaultdict, deque

import numpy as np
from pydantic import BaseModel

from app.config import Settings


def embed_text(text: str, dim: int = 256) -> list[float]:
    v = np.zeros(dim, dtype=np.float32)
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim] += 1.0
    n = np.linalg.norm(v)
    return (v / n if n else v).tolist()


class FakeLLM:
    def __init__(self, script: dict[str, list] | None = None, answer: str = "Answer.",
                 settings: Settings | None = None, token_delay: float = 0.0):
        self.script: dict[str, deque] = defaultdict(deque, {k: deque(v) for k, v in (script or {}).items()})
        self.answer = answer
        self.calls: list[str] = []
        self.settings = settings or Settings().resolved(["qwen2.5:7b-instruct", "nomic-embed-text"])
        self.token_delay = token_delay
        self.last_messages: list[dict] = []

    async def structured(self, messages, schema: type[BaseModel], **_):
        self.calls.append(schema.__name__)
        queue = self.script[schema.__name__]
        item = queue.popleft() if queue else {}
        if isinstance(item, Exception):
            raise item
        if callable(item):
            item = item(messages)
        return schema.model_validate(item)

    async def stream(self, messages, **_):
        self.calls.append("stream")
        self.last_messages = messages
        for word in self.answer.split(" "):
            if self.token_delay:
                await asyncio.sleep(self.token_delay)
            yield word + " "

    async def embed(self, texts):
        return [embed_text(t) for t in texts]

    async def list_models(self):
        return ["qwen2.5:7b-instruct", "nomic-embed-text:latest"]

    async def aclose(self):
        pass

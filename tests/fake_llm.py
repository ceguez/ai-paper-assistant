"""Deterministic stand-in for a local model: scripted structured outputs per schema,
word-streamed answers, and hashed bag-of-words embeddings (so retrieval actually works)."""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict, deque

import numpy as np
from pydantic import BaseModel


def _embed(text: str, dim: int = 256) -> list[float]:
    v = np.zeros(dim, dtype=np.float32)
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim] += 1.0
    n = np.linalg.norm(v)
    return (v / n if n else v).tolist()


class FakeLLM:
    def __init__(self, script: dict[str, list] | None = None, answer: str = "Answer."):
        self.script: dict[str, deque] = defaultdict(deque, {k: deque(v) for k, v in (script or {}).items()})
        self.answer = answer
        self.calls: list[str] = []

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
        for word in self.answer.split(" "):
            yield word + " "

    async def embed(self, texts):
        return [_embed(t) for t in texts]

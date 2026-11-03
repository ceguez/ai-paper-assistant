"""Thin async client for a local model runtime (Ollama native API).

Design notes
- One `LLM` protocol so the orchestrator never knows which runtime is behind it
  (Ollama, llama.cpp server, or the FakeLLM used in tests).
- `structured()` = grammar-constrained decoding (JSON schema in `format`) + Pydantic
  validation + one re-ask carrying the validation error. Constrained decoding
  guarantees shape; validators check meaning.
- num_ctx is ALWAYS sent explicitly: Ollama's default is small and it truncates silently.
"""
from __future__ import annotations

import asyncio
import json
import random
from typing import AsyncIterator, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from .config import settings

T = TypeVar("T", bound=BaseModel)
Message = dict[str, str]


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    async def structured(self, messages: list[Message], schema: type[T], *, model: str | None = None,
                         temperature: float = 0.0, max_tokens: int = 512) -> T: ...

    def stream(self, messages: list[Message], *, temperature: float = 0.5,
               max_tokens: int = 1200) -> AsyncIterator[str]: ...

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class OllamaLLM:
    def __init__(self, base_url: str = settings.ollama_url, timeout: float = 120.0):
        import httpx  # lazy: keeps the rest of the package importable without network deps

        self._http = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    def _options(self, temperature: float, max_tokens: int) -> dict:
        return {"temperature": temperature, "num_predict": max_tokens,
                "num_ctx": settings.budgets.num_ctx}

    async def _post(self, path: str, payload: dict, attempts: int = 3) -> dict:
        """POST with exponential backoff + jitter for transient runtime errors."""
        last: Exception | None = None
        for i in range(attempts):
            try:
                r = await self._http.post(path, json=payload)
                r.raise_for_status()
                return r.json()
            except Exception as e:  # httpx errors, 5xx, model loading
                last = e
                await asyncio.sleep((2 ** i) * 0.5 + random.random() * 0.25)
        raise LLMError(f"{path} failed after {attempts} attempts: {last}")

    async def structured(self, messages, schema, *, model=None, temperature=0.0, max_tokens=512):
        msgs = list(messages)
        for attempt in range(settings.schema_retries + 1):
            data = await self._post("/api/chat", {
                "model": model or settings.gen_model,
                "messages": msgs,
                "format": schema.model_json_schema(),  # constrained decoding
                "stream": False,
                "keep_alive": settings.keep_alive,
                "options": self._options(temperature, max_tokens),
            })
            raw = data["message"]["content"]
            try:
                return schema.model_validate_json(raw)
            except ValidationError as e:
                if attempt == settings.schema_retries:
                    raise LLMError(f"schema {schema.__name__} invalid after retries: {e}") from e
                # Re-ask: show the model its own output and the exact validation error.
                msgs = msgs + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"That JSON failed validation:\n{e}\nReturn corrected JSON only."},
                ]
        raise AssertionError("unreachable")

    async def stream(self, messages, *, temperature=0.5, max_tokens=1200):
        payload = {"model": settings.gen_model, "messages": messages, "stream": True,
                   "keep_alive": settings.keep_alive, "options": self._options(temperature, max_tokens)}
        async with self._http.stream("POST", "/api/chat", json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if delta := chunk.get("message", {}).get("content"):
                    yield delta
                if chunk.get("done"):
                    break

    async def list_models(self) -> list[str]:
        """Names of the models Ollama has pulled (GET /api/tags)."""
        r = await self._http.get("/api/tags", timeout=5.0)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    async def embed(self, texts):
        data = await self._post("/api/embed", {"model": settings.embed_model, "input": texts})
        return data["embeddings"]

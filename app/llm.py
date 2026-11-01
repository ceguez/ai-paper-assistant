"""Local model runtimes behind one interface.

- OllamaLLM: native API. JSON-schema constrained decoding via `format`.
- OpenAICompatLLM: LM Studio or `mlx_lm.server` (Apple's MLX) via /v1. Uses
  `response_format: json_schema` where the server supports it; validation + re-ask still
  catches servers that ignore it.

Both: structured() = constrained decoding -> Pydantic validation -> one re-ask carrying the
validation error. num_ctx is always explicit, because Ollama truncates silently otherwise.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from typing import AsyncIterator, Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .config import Settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
Message = dict[str, str]
FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.S)


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    settings: Settings

    async def structured(self, messages: list[Message], schema: type[T], *, model: str | None = None,
                         temperature: float = 0.0, max_tokens: int = 512) -> T: ...

    def stream(self, messages: list[Message], *, temperature: float = 0.5,
               max_tokens: int = 1200) -> AsyncIterator[str]: ...

    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    async def list_models(self) -> list[str]: ...

    async def aclose(self) -> None: ...


def _clean_json(raw: str) -> str:
    """Strip code fences and leading chatter some runtimes add without constrained decoding."""
    raw = FENCE.sub("", raw.strip())
    start = min((i for i in (raw.find("{"), raw.find("[")) if i >= 0), default=0)
    return raw[start:]


class _Base:
    def __init__(self, settings: Settings, base_url: str, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings
        self._http = httpx.AsyncClient(base_url=base_url, transport=transport,
                                       timeout=httpx.Timeout(settings.request_timeout, connect=5.0))

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _post(self, path: str, payload: dict, attempts: int = 3) -> dict:
        """POST with exponential backoff + jitter. 4xx errors are not retried (they won't fix themselves)."""
        last: Exception | None = None
        for i in range(attempts):
            try:
                r = await self._http.post(path, json=payload)
                if 400 <= r.status_code < 500:
                    raise LLMError(f"{path} -> {r.status_code}: {r.text[:300]}")
                r.raise_for_status()
                return r.json()
            except LLMError:
                raise
            except Exception as e:  # connection refused, 5xx, model still loading
                last = e
                if i < attempts - 1:
                    await asyncio.sleep((2 ** i) * 0.5 + random.random() * 0.25)
        raise LLMError(f"{path} failed after {attempts} attempts: {last!r}")

    async def _complete_json(self, messages: list[Message], schema: type[BaseModel], model: str,
                             temperature: float, max_tokens: int) -> str:
        raise NotImplementedError

    async def structured(self, messages, schema, *, model=None, temperature=0.0, max_tokens=512):
        msgs = list(messages)
        model = model or self.settings.gen_model
        for attempt in range(self.settings.schema_retries + 1):
            raw = await self._complete_json(msgs, schema, model, temperature, max_tokens)
            try:
                return schema.model_validate_json(_clean_json(raw))
            except ValidationError as e:
                log.info("schema %s failed validation (attempt %d): %s", schema.__name__, attempt, e)
                if attempt == self.settings.schema_retries:
                    raise LLMError(f"schema {schema.__name__} invalid after retries: {e}") from e
                msgs = msgs + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"That JSON failed validation:\n{e}\nReturn corrected JSON only."},
                ]
        raise AssertionError("unreachable")


class OllamaLLM(_Base):
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        super().__init__(settings, settings.base_url or "http://127.0.0.1:11434", transport)

    def _options(self, temperature: float, max_tokens: int) -> dict:
        return {"temperature": temperature, "num_predict": max_tokens, "num_ctx": self.settings.budgets.num_ctx}

    def _extra(self, model: str) -> dict:
        # Qwen3 thinks by default; thinking tokens waste the budget and fight `format`.
        return {"think": False} if model.startswith("qwen3") else {}

    async def _complete_json(self, messages, schema, model, temperature, max_tokens):
        data = await self._post("/api/chat", {
            "model": model, "messages": messages, "stream": False,
            "format": schema.model_json_schema(),          # constrained decoding
            "keep_alive": self.settings.keep_alive,
            "options": self._options(temperature, max_tokens), **self._extra(model),
        })
        if data.get("done_reason") == "length":
            log.warning("structured output hit num_predict=%d for %s", max_tokens, schema.__name__)
        return data["message"]["content"]

    async def stream(self, messages, *, temperature=0.5, max_tokens=1200):
        model = self.settings.gen_model
        payload = {"model": model, "messages": messages, "stream": True, "keep_alive": self.settings.keep_alive,
                   "options": self._options(temperature, max_tokens), **self._extra(model)}
        try:
            async with self._http.stream("POST", "/api/chat", json=payload) as r:
                if r.status_code >= 400:
                    body = (await r.aread()).decode(errors="replace")
                    raise LLMError(f"/api/chat -> {r.status_code}: {body[:300]}")
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if err := chunk.get("error"):
                        raise LLMError(err)
                    if delta := chunk.get("message", {}).get("content"):
                        yield delta
                    if chunk.get("done"):
                        break
        except httpx.HTTPError as e:
            raise LLMError(f"stream failed: {e!r}") from e

    async def embed(self, texts):
        data = await self._post("/api/embed", {"model": self.settings.embed_model, "input": texts,
                                               "keep_alive": self.settings.keep_alive})
        return data["embeddings"]

    async def list_models(self) -> list[str]:
        r = await self._http.get("/api/tags", timeout=5.0)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]


class OpenAICompatLLM(_Base):
    """LM Studio (default :1234) or mlx_lm.server (default :8080)."""

    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        super().__init__(settings, settings.base_url or "http://127.0.0.1:1234", transport)

    async def _complete_json(self, messages, schema, model, temperature, max_tokens):
        data = await self._post("/v1/chat/completions", {
            "model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema(),
                                                "strict": True}},
        })
        return data["choices"][0]["message"]["content"]

    async def stream(self, messages, *, temperature=0.5, max_tokens=1200):
        payload = {"model": self.settings.gen_model, "messages": messages, "temperature": temperature,
                   "max_tokens": max_tokens, "stream": True}
        try:
            async with self._http.stream("POST", "/v1/chat/completions", json=payload) as r:
                if r.status_code >= 400:
                    body = (await r.aread()).decode(errors="replace")
                    raise LLMError(f"/v1/chat/completions -> {r.status_code}: {body[:300]}")
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    choice = json.loads(data)["choices"][0]
                    if delta := (choice.get("delta") or {}).get("content"):
                        yield delta
        except httpx.HTTPError as e:
            raise LLMError(f"stream failed: {e!r}") from e

    async def embed(self, texts):
        data = await self._post("/v1/embeddings", {"model": self.settings.embed_model, "input": texts})
        return [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]

    async def list_models(self) -> list[str]:
        r = await self._http.get("/v1/models", timeout=5.0)
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]


def make_llm(settings: Settings, transport: httpx.AsyncBaseTransport | None = None) -> LLM:
    if settings.runtime == "openai":
        return OpenAICompatLLM(settings, transport)
    return OllamaLLM(settings, transport)

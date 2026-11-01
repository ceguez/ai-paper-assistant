"""Central configuration. Values come from env vars prefixed PA_ (e.g. PA_GEN_MODEL)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory


def _env(name: str, default: str) -> str:
    return os.getenv(f"PA_{name}", default)


@dataclass(frozen=True)
class Budgets:
    """Per-slot token budgets for one model call (see doc section 5)."""
    num_ctx: int = 8192
    system: int = 700
    tools: int = 400
    memory: int = 500
    history: int = 1500
    sources: int = 3000
    output_reserve: int = 1200

    def check(self) -> None:
        used = self.system + self.tools + self.memory + self.history + self.sources + self.output_reserve
        if used > int(self.num_ctx * 0.9):  # 10% headroom for token-estimate error
            raise ValueError(f"budgets ({used}) exceed 90% of num_ctx ({self.num_ctx})")


@dataclass(frozen=True)
class Settings:
    ollama_url: str = _env("OLLAMA_URL", "http://localhost:11434")
    gen_model: str = _env("GEN_MODEL", "qwen2.5:7b-instruct-q4_K_M")  # any 7-9B tool-calling model
    router_model: str = _env("ROUTER_MODEL", "")  # empty -> reuse gen_model at temperature 0
    embed_model: str = _env("EMBED_MODEL", "nomic-embed-text")
    db_path: str = _env("DB_PATH", str(ROOT / "data" / "assistant.db"))  # same file wherever you launch from
    upload_dir: str = _env("UPLOAD_DIR", str(ROOT / "data" / "papers"))
    max_steps: int = int(_env("MAX_STEPS", "6"))
    schema_retries: int = int(_env("SCHEMA_RETRIES", "1"))
    keep_alive: str = _env("KEEP_ALIVE", "30m")
    budgets: Budgets = field(default_factory=Budgets)

    @property
    def router(self) -> str:
        return self.router_model or self.gen_model


settings = Settings()
settings.budgets.check()

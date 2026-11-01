"""Configuration tuned for local hardware.

Values come from (lowest to highest priority): the hardware profile, a `.env` file in the
project root, then real environment variables. Every key is prefixed `PA_`.

Profiles are sized for an Apple M4 Pro with 24 GB unified memory. macOS lets the GPU use
roughly two thirds to three quarters of unified memory, so on 24 GB the model weights, KV
cache and embedding model should stay under about 16 GB, leaving room for the OS, browser
and editor. `estimate_memory_gb()` checks a profile against that ceiling.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> dict[str, str]:
    """Minimal .env reader: KEY=VALUE lines, # comments, optional quotes. No dependency."""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


_DOTENV = _load_dotenv(ROOT / ".env")


def env(name: str, default: str = "") -> str:
    key = f"PA_{name}"
    return os.environ.get(key, _DOTENV.get(key, default))


@dataclass(frozen=True)
class Budgets:
    """Per-slot token budgets for one model call (design doc section 5).

    Token counts are estimated (no tokenizer round-trip), so the slots must leave a
    safety margin below num_ctx; `check()` enforces 10%.
    """
    num_ctx: int = 16384
    system: int = 900
    tools: int = 500
    memory: int = 800
    history: int = 3000
    sources: int = 7800
    output_reserve: int = 1500

    @property
    def used(self) -> int:
        return self.system + self.tools + self.memory + self.history + self.sources + self.output_reserve

    def check(self) -> None:
        if self.used > int(self.num_ctx * 0.9):
            raise ValueError(f"token budgets ({self.used}) exceed 90% of num_ctx ({self.num_ctx})")

    @classmethod
    def for_ctx(cls, num_ctx: int) -> "Budgets":
        """Scale the 16k layout to another window, keeping fixed slots fixed."""
        if num_ctx == 16384:
            return cls()
        scale = num_ctx / 16384
        system, tools = min(900, int(900 * scale ** 0.5)), min(500, int(500 * scale ** 0.5))
        memory, history = int(800 * scale), int(3000 * scale)
        output = max(800, min(2000, int(1500 * scale)))
        sources = int(num_ctx * 0.9) - (system + tools + memory + history + output)  # retrieval gets the rest
        return cls(num_ctx=num_ctx, system=system, tools=tools, memory=memory, history=history,
                   sources=sources, output_reserve=output)


@dataclass(frozen=True)
class ModelSpec:
    """A model choice plus what it costs in memory."""
    name: str
    weights_gb: float          # GGUF file size at the stated quantization
    kv_kb_per_token: float     # fp16 KV cache per token = 2 * layers * kv_heads * head_dim * 2 bytes
    max_ctx: int


# Candidates in preference order for each role. Auto-detection picks the first one that is
# already pulled; `doctor` tells you what to pull when none is. Sizes are Ollama's
# default Q4_K_M tags.
GENERATORS = {
    "qwen2.5:7b-instruct": ModelSpec("qwen2.5:7b-instruct", 4.7, 56, 32768),
    "qwen3:8b": ModelSpec("qwen3:8b", 5.2, 144, 40960),
    "llama3.1:8b": ModelSpec("llama3.1:8b", 4.9, 128, 131072),
    "qwen2.5:14b-instruct": ModelSpec("qwen2.5:14b-instruct", 9.0, 192, 32768),
    "qwen2.5:3b-instruct": ModelSpec("qwen2.5:3b-instruct", 1.9, 36, 32768),
}
EMBEDDERS = {
    "nomic-embed-text": ModelSpec("nomic-embed-text", 0.27, 0, 8192),
    "mxbai-embed-large": ModelSpec("mxbai-embed-large", 0.67, 0, 512),
    "bge-m3": ModelSpec("bge-m3", 1.2, 0, 8192),
}


@dataclass(frozen=True)
class Profile:
    name: str
    description: str
    generators: tuple[str, ...]
    num_ctx: int
    kv_cache_type: str = "q8_0"   # set on the Ollama server: OLLAMA_KV_CACHE_TYPE


PROFILES = {
    "balanced": Profile("balanced", "7-8B at Q4_K_M, 16k context. Default for 24 GB.",
                        ("qwen2.5:7b-instruct", "qwen3:8b", "llama3.1:8b"), 16384),
    "quality": Profile("quality", "14B at Q4_K_M, 8k context. Better reasoning, ~2x slower.",
                       ("qwen2.5:14b-instruct",), 8192),
    "fast": Profile("fast", "3B at Q4_K_M, 16k context. For quick lookups and weak machines.",
                    ("qwen2.5:3b-instruct",), 16384),
}

GPU_BUDGET_GB = float(env("GPU_BUDGET_GB", "16"))  # usable for models on a 24 GB Mac


def estimate_memory_gb(gen: str, embed: str, num_ctx: int, kv_cache_type: str = "q8_0") -> float:
    g = GENERATORS.get(gen)
    e = EMBEDDERS.get(embed)
    if g is None:
        return 0.0
    kv_factor = {"f16": 1.0, "q8_0": 0.5, "q4_0": 0.25}.get(kv_cache_type, 1.0)
    kv_gb = g.kv_kb_per_token * num_ctx * kv_factor / (1024 * 1024)
    overhead = 0.6  # compute buffers, runtime
    return round(g.weights_gb + kv_gb + (e.weights_gb if e else 0.0) + overhead, 1)


@dataclass(frozen=True)
class Settings:
    runtime: str = env("RUNTIME", "ollama")                      # "ollama" | "openai" (LM Studio, mlx_lm.server)
    base_url: str = env("BASE_URL", "")                          # empty -> runtime default
    profile: str = env("PROFILE", "balanced")
    gen_model: str = env("GEN_MODEL", "")                        # empty -> auto-detect from profile
    router_model: str = env("ROUTER_MODEL", "")                  # empty -> gen_model at temperature 0
    embed_model: str = env("EMBED_MODEL", "nomic-embed-text")
    num_ctx: int = int(env("NUM_CTX", "0"))                      # 0 -> profile default
    data_dir: str = env("DATA_DIR", str(ROOT / "data"))
    host: str = env("HOST", "127.0.0.1")                         # local only by default
    port: int = int(env("PORT", "8000"))
    max_upload_mb: int = int(env("MAX_UPLOAD_MB", "100"))
    max_steps: int = int(env("MAX_STEPS", "6"))
    schema_retries: int = int(env("SCHEMA_RETRIES", "1"))
    keep_alive: str = env("KEEP_ALIVE", "30m")
    request_timeout: float = float(env("REQUEST_TIMEOUT", "300"))
    parser: str = env("PARSER", "auto")                          # auto | docling | pymupdf4llm
    memory_extraction: bool = env("MEMORY_EXTRACTION", "true").lower() == "true"
    budgets: Budgets = field(default_factory=Budgets)

    @property
    def db_path(self) -> str:
        return str(Path(self.data_dir) / "assistant.db")

    @property
    def upload_dir(self) -> str:
        return str(Path(self.data_dir) / "papers")

    @property
    def router(self) -> str:
        return self.router_model or self.gen_model

    @property
    def active_profile(self) -> Profile:
        return PROFILES.get(self.profile, PROFILES["balanced"])

    def resolved(self, available: list[str] | None = None) -> "Settings":
        """Fill gen_model and num_ctx from the profile and the models actually pulled."""
        prof = self.active_profile
        gen = self.gen_model
        if not gen:
            gen = prof.generators[0]
            if available:
                pulled = {_base(m) for m in available}
                gen = next((g for g in prof.generators if _base(g) in pulled), gen)
        ctx = self.num_ctx or prof.num_ctx
        spec = GENERATORS.get(gen)
        if spec:
            ctx = min(ctx, spec.max_ctx)
        budgets = Budgets.for_ctx(ctx)
        budgets.check()
        return replace(self, gen_model=gen, num_ctx=ctx, budgets=budgets)


def _base(model: str) -> str:
    """'qwen2.5:7b-instruct-q4_K_M' and 'qwen2.5:7b-instruct' match; ':latest' is dropped."""
    name = model.removesuffix(":latest")
    for suffix in ("-q4_K_M", "-q4_k_m", "-q5_K_M", "-q8_0"):
        name = name.removesuffix(suffix)
    return name


def matches(model: str, available: list[str]) -> bool:
    return _base(model) in {_base(m) for m in available}


settings = Settings().resolved()

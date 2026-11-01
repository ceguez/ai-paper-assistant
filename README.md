# Paper Assistant

A research paper assistant that runs entirely on your Mac. Add PDFs, ask questions, and get
answers that cite the exact sections they used, with the page to check. The model, the index
and your notes never leave the machine.

Tuned for an **Apple M4 Pro with 24 GB unified memory**: a 7B model at Q4_K_M with a 16k
context and a local embedding model use about 6 GB, well inside the roughly 16 GB macOS lets
the GPU use.

## Quick start

```bash
brew install uv                 # if needed; Ollama: https://ollama.com/download
cd ai-paper-assistant
bash scripts/setup_mac.sh       # Python 3.12 env, pulls qwen2.5:7b-instruct + nomic-embed-text, runs checks
bash scripts/tune_ollama.sh     # optional, recommended: then quit and reopen Ollama
uv run python -m app serve      # open http://127.0.0.1:8000
```

The UI ships prebuilt in `frontend/dist`, so no Node is needed to run it.

## Hardware profiles

Set `PA_PROFILE` in `.env`. The app picks the first model of the profile that you have
already pulled; `uv run python -m app doctor` tells you what to pull otherwise.

| Profile | Model (Q4_K_M) | Context | Est. memory | Use when |
| --- | --- | --- | --- | --- |
| `balanced` (default) | qwen2.5:7b-instruct, qwen3:8b or llama3.1:8b | 16k | ~6 GB | Everyday use |
| `quality` | qwen2.5:14b-instruct | 8k | ~10.5 GB | Harder comparisons; about half the speed |
| `fast` | qwen2.5:3b-instruct | 16k | ~3 GB | Quick lookups, other apps running |

Estimates cover weights, a q8_0 KV cache at the full context, the embedding model and runtime
overhead. `scripts/tune_ollama.sh` sets the Ollama Mac app's flash attention and q8_0 KV cache, which
halve cache memory, and keeps both models loaded so ingestion and chat don't swap them.

**LM Studio or MLX instead of Ollama:** set `PA_RUNTIME=openai` and `PA_BASE_URL` to
`http://127.0.0.1:1234` (LM Studio) or `http://127.0.0.1:8080` (`mlx_lm.server`), and set
`PA_GEN_MODEL` / `PA_EMBED_MODEL` to the names that server lists.

## What happens to a paper

1. The PDF is hashed (re-uploading the same file is a no-op) and queued. One paper is
   processed at a time so ingestion never fights chat for the GPU.
2. PyMuPDF4LLM (or Docling: `bash scripts/setup_mac.sh --docling`) converts it to
   Markdown with a marker per page.
3. Headings become a section tree; tables and figure captions become their own nodes;
   acknowledgements and references are kept but not indexed.
4. Each section gets a short *pointer* (its path in the paper plus its first lines). Only
   pointers are embedded; at question time a hit loads the **whole section**, so the model
   reads intact arguments rather than fragments (Proxy-Pointer retrieval).
5. Everything commits in one SQLite transaction: the paper, the tree, BM25 keyword index and
   vectors. Search fuses keyword and vector ranks.

## How a question is answered

`route` → `retrieve` → `research` (the model picks tools, at most 6 steps, repeats stop the
loop) → `answer` (streamed, cited) → `verify` (claims checked against the sources). Any
failure drops to a plain search-and-answer path, so you always get a grounded reply.
Runs continue if you close the tab; reopening the chat replays them. The **Research trail**
tab shows every step.

Tick papers in the library to limit answers to them.

## Commands

| Command | What it does |
| --- | --- |
| `uv run python -m app serve` | Run the app on 127.0.0.1:8000 (local only); add `--reload` while editing |
| `uv run python -m app doctor` | Check Ollama, models, memory fit and the parser |
| `uv run --group dev pytest -q` | Backend tests; no model needed |
| `uv run python -m app ingest ~/Papers` | Ingest a folder from the terminal |
| `uv run python -m app ask "question"` | Ask from the terminal, streamed |
| `uv run python -m app reindex` | Re-read every paper (after changing the embedding model) |
| `bash scripts/mock.sh` | Run against a fake Ollama, for UI work without a model |
| `cd frontend && npm install && npm run build` | Rebuild the UI (Node 20+); `npm run dev` rebuilds on change |

## Layout

```
app/
  config.py        profiles, model auto-detection, memory estimate, token budgets
  llm.py           Ollama and OpenAI-compatible clients: schema-constrained JSON, re-ask, streaming
  db.py            SQLite: papers, section tree, FTS5 + vector search, runs, events, memory
  jobs.py          persistent ingestion queue (survives restarts)
  ingest/          parser (page markers), section tree, pipeline
  agent/           state, context budgets, tools, orchestrator + run hub
  memory.py        episodes, fact extraction, incremental session summaries
  api.py           HTTP API + SSE, serves the UI
  __main__.py      serve / doctor / ingest / ask / reindex
frontend/          React 19 + Tailwind 4, built with esbuild (build.mjs); dist/ is prebuilt
scripts/           setup_mac.sh, tune_ollama.sh, mock.sh, mock_ollama.py
tests/             unit, HTTP end-to-end, and Ollama wire-protocol tests
```

## Notes

- PyMuPDF is AGPL-licensed: fine for personal and internal use; check before redistributing.
- Scanned PDFs without a text layer produce no text; run OCR first (e.g. `ocrmypdf`).
- Data lives in `./data` (`PA_DATA_DIR`). Back it up or delete it to start over.

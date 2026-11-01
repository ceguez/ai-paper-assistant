# Research Paper Assistant — backend skeleton

Local-first agent over your paper library: Proxy-Pointer retrieval, a native async state
machine, and one SQLite file for state, memory and vectors. Companion to the design doc.

## Layout

```
app/
  config.py          settings + per-slot token budgets
  llm.py             Ollama client: constrained JSON (schema in `format`) + validate + re-ask; streaming; embeddings
  db.py              SQLite schema, hybrid search (FTS5 BM25 + cosine, RRF fusion), checkpoints, events, memory
  ingest/
    parser.py        PDF -> Markdown (Docling, PyMuPDF4LLM fallback)
    tree.py          skeleton tree, table/figure nodes, node sizing, pointer text (pure Python)
    pipeline.py      fingerprint -> parse -> metadata -> tree -> pointers -> embed -> atomic commit
  agent/
    state.py         RunState + every schema the model is constrained to
    context.py       ContextAssembler: fixed slot order, hard token budgets
    tools.py         search_papers, read_section (pointer -> whole section), get_element, save_note
    graph.py         Orchestrator: route -> retrieve -> research loop -> answer -> verify; fallback
  memory.py          post-run: episodes, fact extraction with invalidation, session compaction
  api.py             FastAPI + SSE event stream
tests/               FakeLLM + end-to-end tests (no model needed)
```

## Run

```bash
pip install -r requirements.txt
ollama pull qwen2.5:7b-instruct-q4_K_M   # or any 7-9B tool-calling model; set PA_GEN_MODEL
ollama pull nomic-embed-text
export OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0
uvicorn app.api:app --reload
pytest -q                                # runs without Ollama
```

## Try it

```bash
curl -F file=@paper.pdf localhost:8000/papers
SID=$(curl -s -X POST localhost:8000/sessions | jq -r .session_id)
RID=$(curl -s -X POST localhost:8000/sessions/$SID/runs -H 'content-type: application/json' \
      -d '{"message":"What learning rate do they use?"}' | jq -r .run_id)
curl -N localhost:8000/runs/$RID/events
```

## Swap points

- Vector index: replace the numpy scan in `Store.search` with sqlite-vec, pgvector or FAISS.
- Tokenizer: replace `count_tokens` in `ingest/tree.py` with the model's tokenizer.
- Runtime: implement the `LLM` protocol for llama.cpp server or any OpenAI-compatible endpoint.
- Jobs: move ingestion from FastAPI `BackgroundTasks` to a queue (arq, RQ) for multi-user.

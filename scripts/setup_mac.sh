#!/usr/bin/env bash
# One-time setup on Apple Silicon. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf "\n\033[1m%s\033[0m\n" "$1"; }

[[ "$(uname -m)" == "arm64" ]] || echo "Note: tuned for Apple Silicon; continuing anyway."

say "1/4 Checking tools"
if ! command -v uv >/dev/null; then
  echo "uv is not installed. Install it with:  brew install uv"
  exit 1
fi
if ! command -v ollama >/dev/null; then
  echo "Ollama is not installed. Get the Mac app from https://ollama.com/download (or: brew install ollama)."
  exit 1
fi
echo "uv $(uv --version | awk '{print $2}'), ollama $(ollama --version 2>/dev/null | awk '{print $NF}')"

say "2/4 Installing Python dependencies (Python 3.12 via uv)"
if [[ "${1:-}" == "--docling" ]]; then
  uv sync --extra docling
else
  uv sync
fi

say "3/4 Pulling models (about 5 GB the first time)"
if ! curl -sf http://127.0.0.1:11434/api/version >/dev/null; then
  echo "Ollama isn't running. Open the Ollama app (or run 'ollama serve' in another terminal), then re-run: bash scripts/setup_mac.sh"
  exit 1
fi
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text

[[ -f .env ]] || cp .env.example .env

say "4/4 Checking the setup"
uv run python -m app doctor || true

cat <<'MSG'

Optional, recommended once: `bash scripts/tune_ollama.sh`, then quit and reopen Ollama.
It halves KV-cache memory and keeps the chat and embedding models loaded together.

Start the app:  uv run python -m app serve   ->  http://127.0.0.1:8000
MSG

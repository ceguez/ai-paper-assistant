#!/usr/bin/env bash
# Run the app against a fake Ollama (for UI work without loading a model). Ctrl+C stops both.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python scripts/mock_ollama.py --port 11435 &
MOCK=$!
trap 'kill $MOCK 2>/dev/null' EXIT
PA_BASE_URL=http://127.0.0.1:11435 PA_DATA_DIR=./data-mock uv run python -m app serve

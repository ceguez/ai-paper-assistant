#!/usr/bin/env bash
# Create .venv with a native Python 3.11-3.13 and install the dependencies.
#   bash scripts/setup_venv.sh             core runtime
#   bash scripts/setup_venv.sh --dev       + pytest
#   bash scripts/setup_venv.sh --docling   + Docling (large download)
# Safe to re-run: an existing .venv is reused and only missing packages are installed.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=""
for c in python3.12 python3.13 python3.11 python3; do
  if command -v "$c" >/dev/null 2>&1 &&
     "$c" -c 'import sys; sys.exit(not (3, 11) <= sys.version_info[:2] <= (3, 13))'; then
    PY=$(command -v "$c"); break
  fi
done
if [[ -z "$PY" ]]; then
  echo "Need Python 3.11-3.13 (3.12 recommended). On a Mac: brew install python@3.12"; exit 1
fi

# A venv costs nothing at runtime, but an Intel (x86_64) Python on Apple Silicon runs under
# Rosetta translation: PDF parsing and NumPy get several times slower, and PyTorch loses MPS.
ARCH=$("$PY" -c 'import platform; print(platform.machine())')
if [[ "$(uname -s)" == "Darwin" && "$ARCH" != "arm64" ]]; then
  echo "$PY is an $ARCH build, which runs under Rosetta on Apple Silicon."
  echo "Install a native one (Homebrew in /opt/homebrew): brew install python@3.12, then re-run."
  exit 1
fi

[[ -d .venv ]] || "$PY" -m venv .venv
.venv/bin/python -m pip install --quiet --upgrade pip

REQS=(-r requirements.txt)
for arg in "$@"; do
  case "$arg" in
    --dev) REQS+=(-r requirements-dev.txt) ;;
    --docling) REQS+=(-r requirements-docling.txt) ;;
    *) echo "unknown option: $arg"; exit 2 ;;
  esac
done
.venv/bin/python -m pip install "${REQS[@]}"

echo
echo "Ready: $(.venv/bin/python -V) ($ARCH) in .venv"
echo "Activate:  source .venv/bin/activate"
echo "Then:      uvicorn app.api:app        # http://127.0.0.1:8000"
echo "           python scripts/ingest.py pdfDocs"

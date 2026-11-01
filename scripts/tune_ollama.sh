#!/usr/bin/env bash
# Memory settings for the Ollama Mac app. Quit and reopen Ollama afterwards for them to apply.
# Undo with: launchctl unsetenv OLLAMA_FLASH_ATTENTION (and the same for each name below).
set -euo pipefail
launchctl setenv OLLAMA_FLASH_ATTENTION 1       # required for a quantized KV cache
launchctl setenv OLLAMA_KV_CACHE_TYPE q8_0      # halves KV-cache memory, negligible quality loss
launchctl setenv OLLAMA_NUM_PARALLEL 1          # one request at a time: predictable memory on 24 GB
launchctl setenv OLLAMA_MAX_LOADED_MODELS 2     # chat model + embedding model stay loaded together
launchctl setenv OLLAMA_KEEP_ALIVE 30m
echo "Set. Quit Ollama from the menu bar and open it again for these to apply."

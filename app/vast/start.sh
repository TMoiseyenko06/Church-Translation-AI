#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start.sh – Vast AI worker (Whisper + Ollama/Qwen2.5 + XTTS v2)
#
# Usage:
#   chmod +x start.sh && ./start.sh
#
# What this script does:
#   1. Verifies Ollama is running and the model is pulled
#   2. Starts the FastAPI worker on port 8001
#
# Prerequisites (run once before first start — see full instructions):
#   • CUDA driver + CUDA toolkit installed
#   • PyTorch (CUDA build) installed
#   • pip install -r requirements.txt
#   • Ollama installed + qwen2.5:7b pulled
#   • ffmpeg on PATH
#   • Port 8001 open in Vast instance TCP port settings
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PORT=8001
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:7b}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"

# ── 1. Check Ollama is reachable ──────────────────────────────────────────────
echo ""
echo "  Checking Ollama …"
if ! curl -sf "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
  echo ""
  echo "  ERROR: Ollama is not running."
  echo "  Start it in a separate terminal with:  ollama serve"
  echo "  Then pull the model with:              ollama pull ${OLLAMA_MODEL}"
  echo ""
  exit 1
fi

# Check the model exists locally.
if ! curl -sf "${OLLAMA_URL}/api/tags" | grep -q "${OLLAMA_MODEL}"; then
  echo "  Model '${OLLAMA_MODEL}' not found locally — pulling now …"
  ollama pull "${OLLAMA_MODEL}"
fi

echo "  Ollama OK (model: ${OLLAMA_MODEL})"

# ── 2. Print connection hint ──────────────────────────────────────────────────
PUBLIC_IP=$(curl -s --max-time 5 https://api.ipify.org 2>/dev/null || echo "unknown")

echo ""
echo "  ┌─────────────────────────────────────────────────────────────┐"
echo "  │  Church Translation – Vast AI Worker                        │"
echo "  │  Listening on 0.0.0.0:${PORT}                                  │"
echo "  │                                                             │"
echo "  │  Tell the VPS:                                              │"
echo "  │    export VAST_WS_URL=ws://${PUBLIC_IP}:${PORT}/ws/worker   │"
echo "  └─────────────────────────────────────────────────────────────┘"
echo ""

# ── 3. Start the FastAPI worker ───────────────────────────────────────────────
OLLAMA_URL="${OLLAMA_URL}" \
OLLAMA_MODEL="${OLLAMA_MODEL}" \
  uvicorn worker:app --host 0.0.0.0 --port "${PORT}" --reload false

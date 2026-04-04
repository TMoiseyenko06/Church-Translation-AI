#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start.sh – Vast AI worker (Whisper + Ollama/Qwen2.5 + XTTS v2)
#
# Usage:
#   chmod +x start.sh && ./start.sh
#
# What this script does:
#   1. Activates the "church" conda environment (Python 3.11)
#   2. Verifies Ollama is running and the model is pulled
#   3. Starts the FastAPI worker on port 8001
#
# One-time setup (run once before first start):
#   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
#   bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
#   source $HOME/miniconda3/etc/profile.d/conda.sh
#   conda create -n church python=3.11 -y
#   conda activate church
#   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
#   pip install -r requirements.txt
#   ollama pull qwen2.5:14b
#   • ffmpeg on PATH:  sudo apt install -y ffmpeg
#   • Port 8001 open in Vast instance TCP port settings
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── 0. Activate conda environment ────────────────────────────────────────────
CONDA_ENV="${CONDA_ENV:-church}"
MINICONDA_PATH="${MINICONDA_PATH:-$HOME/miniconda3}"

if [ -f "${MINICONDA_PATH}/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "${MINICONDA_PATH}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
  echo "  Activated conda env: ${CONDA_ENV} ($(python --version))"
else
  echo ""
  echo "  WARNING: conda not found at ${MINICONDA_PATH}."
  echo "  Continuing with system Python — install may fail if not Python 3.11."
  echo "  Set MINICONDA_PATH if conda is installed elsewhere."
  echo ""
fi

PORT=8001
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:14b}"
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
  uvicorn worker:app --host 0.0.0.0 --port "${PORT}"

#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start.sh – Vast AI worker (Whisper + Ollama/Qwen2.5 + XTTS v2)
#
# Usage:
#   chmod +x start.sh && ./start.sh
#
# What this script does:
#   1. Activates the "church" conda environment (Python 3.11)
#   2. Starts Ollama in the background if not already running
#   3. Pulls the model if not already downloaded
#   4. Starts the FastAPI worker on port 8001
#
# One-time setup (run once before first start):
#   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
#   bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
#   source $HOME/miniconda3/etc/profile.d/conda.sh
#   conda create -n church python=3.11 -y
#   conda activate church
#   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
#   pip install -r requirements.txt
#   • ffmpeg on PATH:  sudo apt install -y ffmpeg
#   • Port 8001 open in Vast instance TCP port settings
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── 0. Activate conda environment ─────────────────────────────────────────────
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

PORT=8888
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:32b}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
OLLAMA_LOG="/tmp/ollama-church.log"

# ── 1. Start Ollama if not already running ────────────────────────────────────
echo ""
if curl -sf "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
  echo "  Ollama already running."
else
  echo "  Starting Ollama in background …"
  nohup ollama serve >"${OLLAMA_LOG}" 2>&1 &
  OLLAMA_PID=$!
  echo "  Ollama PID: ${OLLAMA_PID} (logs: ${OLLAMA_LOG})"

  # Wait up to 15 seconds for Ollama to become ready.
  for i in $(seq 1 15); do
    if curl -sf "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
      echo "  Ollama ready."
      break
    fi
    if [ "$i" -eq 15 ]; then
      echo "  ERROR: Ollama did not start in time. Check ${OLLAMA_LOG}"
      exit 1
    fi
    sleep 1
  done
fi

# ── 2. Delete all other models, then pull the target model ───────────────────
echo "  Removing any existing models …"
EXISTING=$(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}' | grep -v '^$' || true)
for model in $EXISTING; do
  if [ "$model" != "${OLLAMA_MODEL}" ]; then
    echo "  Deleting: $model"
    ollama rm "$model" 2>/dev/null || true
  fi
done

if ! ollama list 2>/dev/null | grep -q "^${OLLAMA_MODEL}"; then
  echo "  Model '${OLLAMA_MODEL}' not found — pulling now (~19 GB, please wait) …"
  ollama pull "${OLLAMA_MODEL}"
else
  echo "  Model '${OLLAMA_MODEL}' already present."
fi

echo "  Ollama OK (model: ${OLLAMA_MODEL})"

# ── 3. Print connection hint ──────────────────────────────────────────────────
# Vast injects VAST_TCP_PORT_X with the external mapped port for internal port X.
# Fall back to the internal port if the variable is not set (e.g. local testing).
PUBLIC_IP=$(curl -s --max-time 5 https://api.ipify.org 2>/dev/null || echo "unknown")
EXTERNAL_PORT="${VAST_TCP_PORT_8888:-${PORT}}"
VAST_WS_URL="ws://${PUBLIC_IP}:${EXTERNAL_PORT}/ws/worker"

echo ""
echo "  ┌─────────────────────────────────────────────────────────────┐"
echo "  │  Church Translation – Vast AI Worker                        │"
echo "  │  Internal port : ${PORT}                                       │"
echo "  │  External port : ${EXTERNAL_PORT}                                   │"
echo "  │                                                             │"
echo "  │  Copy this to your VPS:                                     │"
echo "  │    export VAST_WS_URL=${VAST_WS_URL}"
echo "  └─────────────────────────────────────────────────────────────┘"
echo ""

# ── 4. Start the FastAPI worker ───────────────────────────────────────────────
# Trap Ctrl-C to also stop the background Ollama process we may have started.
trap 'echo ""; echo "  Shutting down …"; kill $(pgrep -f "ollama serve") 2>/dev/null || true; exit 0' INT TERM

OLLAMA_URL="${OLLAMA_URL}" \
OLLAMA_MODEL="${OLLAMA_MODEL}" \
  uvicorn worker:app --host 0.0.0.0 --port "${PORT}"

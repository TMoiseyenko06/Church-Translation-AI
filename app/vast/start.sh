#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start.sh – Vast AI worker (Whisper + NLLB translation + edge-tts)
#
# Usage:
#   chmod +x start.sh && ./start.sh
#
# One-time setup:
#   conda create -n church python=3.11 -y && conda activate church
#   pip install torch --index-url https://download.pytorch.org/whl/cu124
#   pip install -r requirements.txt
#   sudo apt install -y ffmpeg
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── 0. Activate conda environment ─────────────────────────────────────────────
CONDA_ENV="${CONDA_ENV:-church}"
MINICONDA_PATH="${MINICONDA_PATH:-$HOME/miniconda3}"

if [ -f "${MINICONDA_PATH}/etc/profile.d/conda.sh" ]; then
  source "${MINICONDA_PATH}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
  echo "  Activated conda env: ${CONDA_ENV} ($(python --version))"
else
  echo "  WARNING: conda not found at ${MINICONDA_PATH}. Using system Python."
fi

# ── CUDA library path (required for faster-whisper int8_float16) ──────────────
CUDA_LIB="${MINICONDA_PATH}/envs/${CONDA_ENV}/lib/python3.11/site-packages/nvidia/cublas/lib"
if [ -d "${CUDA_LIB}" ]; then
  export LD_LIBRARY_PATH="${CUDA_LIB}:${LD_LIBRARY_PATH:-}"
fi

PORT=8888

# ── Print connection hint ──────────────────────────────────────────────────────
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

# ── Start the FastAPI worker ──────────────────────────────────────────────────
trap 'echo "Shutting down …"; exit 0' INT TERM
uvicorn worker:app --host 0.0.0.0 --port "${PORT}"

#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-church}"
MINICONDA_PATH="${MINICONDA_PATH:-$HOME/miniconda3}"

if [ -f "${MINICONDA_PATH}/etc/profile.d/conda.sh" ]; then
  source "${MINICONDA_PATH}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
  echo "  Activated conda env: ${CONDA_ENV} ($(python --version))"
else
  echo "  WARNING: conda not found at ${MINICONDA_PATH}. Using system Python."
fi

# CUDA library path (required for faster-whisper int8_float16)
CUDA_LIB="${MINICONDA_PATH}/envs/${CONDA_ENV}/lib/python3.11/site-packages/nvidia/cublas/lib"
if [ -d "${CUDA_LIB}" ]; then
  export LD_LIBRARY_PATH="${CUDA_LIB}:${LD_LIBRARY_PATH:-}"
fi

PORT=8888
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:7b}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
OLLAMA_LOG="/tmp/ollama-church.log"

# Start Ollama if not already running
if curl -sf "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
  echo "  Ollama already running."
else
  echo "  Starting Ollama in background …"
  nohup ollama serve >"${OLLAMA_LOG}" 2>&1 &
  for i in $(seq 1 15); do
    curl -sf "${OLLAMA_URL}/api/tags" >/dev/null 2>&1 && echo "  Ollama ready." && break
    [ "$i" -eq 15 ] && echo "  ERROR: Ollama did not start. Check ${OLLAMA_LOG}" && exit 1
    sleep 1
  done
fi

# Delete all models except the target, then pull if missing
echo "  Removing other models …"
for model in $(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}' | grep -v '^$'); do
  if [ "$model" != "${OLLAMA_MODEL}" ]; then
    echo "  Deleting: $model"
    ollama rm "$model" 2>/dev/null || true
  fi
done

# Pull model if missing
if ! ollama list 2>/dev/null | grep -q "^${OLLAMA_MODEL}"; then
  echo "  Pulling '${OLLAMA_MODEL}' …"
  ollama pull "${OLLAMA_MODEL}"
fi
echo "  Ollama OK (model: ${OLLAMA_MODEL})"

# Print connection info
PUBLIC_IP=$(curl -s --max-time 5 https://api.ipify.org 2>/dev/null || echo "unknown")
EXTERNAL_PORT="${VAST_TCP_PORT_8888:-${PORT}}"
echo ""
echo "  VAST_WS_URL=ws://${PUBLIC_IP}:${EXTERNAL_PORT}/ws/worker"
echo ""

trap 'kill $(pgrep -f "ollama serve") 2>/dev/null || true; exit 0' INT TERM
OLLAMA_URL="${OLLAMA_URL}" OLLAMA_MODEL="${OLLAMA_MODEL}" \
  uvicorn worker:app --host 0.0.0.0 --port "${PORT}"

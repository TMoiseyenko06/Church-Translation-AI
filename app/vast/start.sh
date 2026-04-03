#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start.sh – Vast AI worker
#
# Usage:
#   chmod +x start.sh && ./start.sh
#
# Requires:
#   • CUDA-capable GPU
#   • Python 3.10+ with Vast requirements installed
#       pip install -r requirements.txt
#   • ffmpeg on PATH
#   • Port 8001 accessible from the VPS (open in Vast instance firewall rules)
#
# The VPS relay connects to this server at:
#   ws://<this-machine-public-ip>:8001/ws/worker
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PORT=8001

echo ""
echo "  ┌─────────────────────────────────────────────────────────┐"
echo "  │  Church Translation – Vast AI Worker                    │"
echo "  │  Listening on 0.0.0.0:${PORT}                              │"
echo "  │  VPS should connect to: ws://<THIS_IP>:${PORT}/ws/worker  │"
echo "  └─────────────────────────────────────────────────────────┘"
echo ""

# Print this machine's public IP as a convenience hint.
PUBLIC_IP=$(curl -s --max-time 5 https://api.ipify.org 2>/dev/null || echo "unknown")
echo "  Detected public IP: ${PUBLIC_IP}"
echo "  → Tell the VPS:  export VAST_WS_URL=ws://${PUBLIC_IP}:${PORT}/ws/worker"
echo ""

uvicorn worker:app --host 0.0.0.0 --port "${PORT}" --reload false

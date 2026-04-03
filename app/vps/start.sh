#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start.sh – VPS relay server + Cloudflare Tunnel
#
# Usage:
#   export VAST_WS_URL=ws://<your-vast-ip>:8001/ws/worker
#   chmod +x start.sh && ./start.sh
#
# Requires:
#   • Python 3.10+ with VPS requirements installed
#       pip install -r requirements.txt
#   • cloudflared on PATH
#       https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PORT=8000
CLOUDFLARED_LOG="/tmp/cloudflared-church-$(date +%s).log"

# ── Validate required environment variable ────────────────────────────────────
if [[ -z "${VAST_WS_URL:-}" ]]; then
  echo ""
  echo "  ERROR: VAST_WS_URL is not set."
  echo "  Set it to the WebSocket address of your Vast AI worker, e.g.:"
  echo "    export VAST_WS_URL=ws://12.34.56.78:8001/ws/worker"
  echo ""
  exit 1
fi

echo ""
echo "  Vast worker target: ${VAST_WS_URL}"

# ── 1. Start the FastAPI relay server ─────────────────────────────────────────
echo "  Starting VPS relay server on port ${PORT} …"
VAST_WS_URL="${VAST_WS_URL}" uvicorn main:app --host 0.0.0.0 --port "${PORT}" &
UVICORN_PID=$!

sleep 2

# ── 2. Start cloudflared quick-tunnel ─────────────────────────────────────────
echo "  Starting Cloudflare Tunnel …"
cloudflared tunnel --url "http://localhost:${PORT}" >"${CLOUDFLARED_LOG}" 2>&1 &
CLOUDFLARED_PID=$!

# ── 3. Wait for the public tunnel URL ─────────────────────────────────────────
echo "  Waiting for tunnel URL (may take 10–20 seconds) …"
TUNNEL_URL=""
for i in $(seq 1 40); do
    TUNNEL_URL=$(grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' \
                     "${CLOUDFLARED_LOG}" 2>/dev/null | head -1 || true)
    [[ -n "${TUNNEL_URL}" ]] && break
    sleep 1
done

# ── 4. Print URLs ──────────────────────────────────────────────────────────────
echo ""
echo "  ╔══════════════════════════════════════════════════════════╗"
if [[ -n "${TUNNEL_URL}" ]]; then
    echo "  ║  PUBLIC TUNNEL:    ${TUNNEL_URL}"
    echo "  ║"
    echo "  ║  Booth (speaker):  ${TUNNEL_URL}/booth"
    echo "  ║  Listen (audience):${TUNNEL_URL}/listen"
else
    echo "  ║  Tunnel URL not detected – check ${CLOUDFLARED_LOG}"
    echo "  ║  Local fallback:  http://localhost:${PORT}/booth"
    echo "  ║                   http://localhost:${PORT}/listen"
fi
echo "  ╚══════════════════════════════════════════════════════════╝"
echo ""

# ── 5. Wait; clean up on Ctrl-C ───────────────────────────────────────────────
trap 'echo ""; echo "  Shutting down …"; kill ${UVICORN_PID} ${CLOUDFLARED_PID} 2>/dev/null; exit 0' INT TERM
wait "${UVICORN_PID}"
kill "${CLOUDFLARED_PID}" 2>/dev/null || true

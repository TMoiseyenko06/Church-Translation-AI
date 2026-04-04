#!/usr/bin/env bash
set -euo pipefail

PORT=8000
CLOUDFLARED_LOG="/tmp/cloudflared.log"

if [[ -z "${VAST_WS_URL:-}" ]]; then
  echo "ERROR: VAST_WS_URL is not set."
  exit 1
fi

echo "Vast worker: ${VAST_WS_URL}"
echo "Starting relay on port ${PORT} …"

VAST_WS_URL="${VAST_WS_URL}" uvicorn main:app --host 0.0.0.0 --port "${PORT}" &
UVICORN_PID=$!

sleep 2

echo "Starting Cloudflare Tunnel …"
cloudflared tunnel --url "http://localhost:${PORT}" >"${CLOUDFLARED_LOG}" 2>&1 &
CLOUDFLARED_PID=$!

TUNNEL_URL=""
for i in $(seq 1 40); do
    TUNNEL_URL=$(grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' "${CLOUDFLARED_LOG}" 2>/dev/null | head -1 || true)
    [[ -n "${TUNNEL_URL}" ]] && break
    sleep 1
done

echo ""
echo "========================================"
if [[ -n "${TUNNEL_URL}" ]]; then
    echo "  Booth:  ${TUNNEL_URL}/booth"
    echo "  Listen: ${TUNNEL_URL}/listen"
else
    echo "  Tunnel not detected — check ${CLOUDFLARED_LOG}"
    echo "  Direct: http://<vps-ip>:${PORT}/booth"
fi
echo "========================================"
echo ""

trap 'kill ${UVICORN_PID} ${CLOUDFLARED_PID} 2>/dev/null; exit 0' INT TERM
wait "${UVICORN_PID}"
kill "${CLOUDFLARED_PID}" 2>/dev/null || true

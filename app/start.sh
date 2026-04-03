#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start.sh – Launch the Church Translation server + Cloudflare Tunnel
#
# Usage:
#   chmod +x start.sh
#   ./start.sh
#
# Requires:
#   • Python 3.10+ with requirements installed  (pip install -r requirements.txt)
#   • ffmpeg available on PATH
#   • cloudflared available on PATH
#     Install: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

CLOUDFLARED_LOG="/tmp/cloudflared-church-$(date +%s).log"
PORT=8000

# ── 1. Start the FastAPI server in the background ────────────────────────────
echo ""
echo "  Starting FastAPI server on port ${PORT}…"
uvicorn main:app --host 0.0.0.0 --port "${PORT}" &
UVICORN_PID=$!

# Give uvicorn a moment to bind the port before cloudflared probes it.
sleep 2

# ── 2. Start cloudflared quick-tunnel in the background ──────────────────────
echo "  Starting Cloudflare Tunnel…"
cloudflared tunnel --url "http://localhost:${PORT}" >"${CLOUDFLARED_LOG}" 2>&1 &
CLOUDFLARED_PID=$!

# ── 3. Wait for cloudflared to print the public URL ──────────────────────────
echo "  Waiting for tunnel URL (may take 10–20 seconds)…"
TUNNEL_URL=""
for i in $(seq 1 40); do
    # cloudflared prints the URL to stdout in the form:
    #   https://<random-subdomain>.trycloudflare.com
    TUNNEL_URL=$(grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' \
                     "${CLOUDFLARED_LOG}" 2>/dev/null | head -1 || true)

    if [[ -n "${TUNNEL_URL}" ]]; then
        break
    fi
    sleep 1
done

# ── 4. Print the public URLs ──────────────────────────────────────────────────
echo ""
echo "  ╔══════════════════════════════════════════════════════════╗"
if [[ -n "${TUNNEL_URL}" ]]; then
    echo "  ║  PUBLIC TUNNEL:  ${TUNNEL_URL}"
    echo "  ║"
    echo "  ║  Booth (speaker):  ${TUNNEL_URL}/booth"
    echo "  ║  Listen (audience): ${TUNNEL_URL}/listen"
else
    echo "  ║  Could not detect tunnel URL – check ${CLOUDFLARED_LOG}"
    echo "  ║  Local only:  http://localhost:${PORT}/booth"
    echo "  ║               http://localhost:${PORT}/listen"
fi
echo "  ╚══════════════════════════════════════════════════════════╝"
echo ""

# ── 5. Keep running until the user presses Ctrl-C ────────────────────────────
trap 'echo ""; echo "  Shutting down…"; kill ${UVICORN_PID} ${CLOUDFLARED_PID} 2>/dev/null; exit 0' INT TERM

# Wait on the uvicorn process; if it exits we also stop cloudflared.
wait "${UVICORN_PID}"
kill "${CLOUDFLARED_PID}" 2>/dev/null || true

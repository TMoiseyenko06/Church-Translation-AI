"""
Church Translation – VPS Relay Server
======================================
Responsibilities:
  • Serve booth.html and listen.html to browsers
  • Accept WebSocket from the translation booth  (/ws/booth)
  • Accept WebSockets from language listeners     (/ws/listen/{lang})
  • Maintain a persistent WebSocket connection to the Vast AI worker
  • Forward audio chunks + active-language metadata to Vast
  • Receive per-language audio results from Vast and push to listeners
  • Echo transcripts back to the booth UI

NO machine-learning libraries run here – this node is intentionally lightweight.

Framing protocol (VPS ↔ Vast, binary WebSocket frames):
  ┌──────────────────────────────────────────────────────┐
  │  4 bytes (big-endian uint32) = length of JSON header │
  │  N bytes  = UTF-8 JSON header                        │
  │  M bytes  = binary payload (WebM audio / MP3 audio)  │
  └──────────────────────────────────────────────────────┘
"""

import asyncio
import json
import logging
import os
import struct
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Optional, Set

import websockets
import websockets.exceptions
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vps-relay")

# ─── Configuration (override via environment variables) ──────────────────────

# WebSocket URL of the Vast AI worker, e.g. ws://12.34.56.78:8001/ws/worker
VAST_WS_URL: str = os.environ.get("VAST_WS_URL", "ws://localhost:8001/ws/worker")

# ─── Language configuration ───────────────────────────────────────────────────
# The Vast worker translates Russian sermon audio to English only.

SUPPORTED_LANGUAGES: Dict[str, str] = {
    "en": "English",
}

# ─── Global state ─────────────────────────────────────────────────────────────

# Live connection to the Vast worker (None when disconnected / reconnecting).
vast_ws: Optional[websockets.WebSocketClientProtocol] = None  # type: ignore[name-defined]

# Mutex so concurrent coroutines don't interleave sends to the Vast WebSocket.
vast_send_lock = asyncio.Lock()

# Current booth WebSocket (single booth per session).
booth_ws: Optional[WebSocket] = None

# Active listener WebSockets keyed by language code.
listeners: Dict[str, Set[WebSocket]] = {lang: set() for lang in SUPPORTED_LANGUAGES}
listeners_lock = asyncio.Lock()

# ─── Framing helpers ──────────────────────────────────────────────────────────

def pack_message(meta: dict, data: bytes) -> bytes:
    """Encode a (metadata-dict, binary-payload) pair into a single frame."""
    meta_bytes = json.dumps(meta).encode()
    return struct.pack(">I", len(meta_bytes)) + meta_bytes + data


def unpack_message(raw: bytes) -> tuple[dict, bytes]:
    """Decode a raw binary frame back into (metadata-dict, binary-payload)."""
    if len(raw) < 4:
        raise ValueError("Frame too short to contain a length header.")
    json_len = struct.unpack(">I", raw[:4])[0]
    if len(raw) < 4 + json_len:
        raise ValueError("Frame truncated: JSON region incomplete.")
    meta = json.loads(raw[4 : 4 + json_len])
    data = raw[4 + json_len :]
    return meta, data


# ─── Vast connection management ──────────────────────────────────────────────

async def vast_receive_loop(ws: websockets.WebSocketClientProtocol) -> None:  # type: ignore[name-defined]
    """
    Consume result frames from the Vast worker indefinitely.

    Each frame contains:
      meta = {"type": "result", "lang": "en", "transcript": "…", "detected_lang": "en"}
      data = MP3 audio bytes for that language
    """
    global booth_ws

    async for raw_msg in ws:
        if not isinstance(raw_msg, bytes):
            continue  # ignore unexpected text frames

        try:
            meta, audio_bytes = unpack_message(raw_msg)
        except Exception as exc:
            logger.error(f"Failed to unpack Vast frame: {exc}")
            continue

        msg_type = meta.get("type")

        if msg_type == "result":
            lang        = meta.get("lang", "")
            transcript  = meta.get("transcript", "")
            detected    = meta.get("detected_lang", "")

            # Forward the transcript text back to the booth UI (once per chunk,
            # the worker sends the transcript alongside the FIRST language result).
            if transcript and booth_ws:
                try:
                    await booth_ws.send_text(
                        json.dumps({"transcript": transcript, "language": detected})
                    )
                except Exception:
                    pass  # booth may have disconnected between sends

            # Push the synthesised audio to all listeners on this language channel.
            if lang and audio_bytes:
                await push_audio_to_listeners(lang, audio_bytes)

        elif msg_type == "error":
            logger.error(f"Vast worker reported error: {meta.get('message')}")


async def connect_to_vast_loop() -> None:
    """
    Background task: keep a persistent WebSocket connection to the Vast worker.
    Reconnects with exponential back-off (2 s → 4 s → … → 60 s max) on failure.
    """
    global vast_ws
    backoff = 2

    while True:
        try:
            logger.info(f"Connecting to Vast worker at {VAST_WS_URL} …")
            async with websockets.connect(
                VAST_WS_URL,
                ping_interval=20,
                ping_timeout=30,
                max_size=100 * 1024 * 1024,  # 100 MB – large enough for any audio chunk
            ) as ws:
                vast_ws = ws
                backoff = 2  # reset back-off after a successful connection
                logger.info("Connected to Vast worker.")
                await vast_receive_loop(ws)

        except websockets.exceptions.ConnectionClosedOK:
            logger.info("Vast worker closed the connection cleanly.")
        except Exception as exc:
            logger.error(f"Vast connection error: {exc}")
        finally:
            vast_ws = None

        logger.info(f"Reconnecting to Vast in {backoff} s …")
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 60)


# ─── App lifecycle ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(connect_to_vast_loop())
    logger.info("VPS relay server starting.")
    yield
    task.cancel()
    logger.info("VPS relay server stopped.")


app = FastAPI(title="Church Translation VPS Relay", lifespan=lifespan)

# ─── Static page routes ───────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.get("/booth")
async def booth_page():
    return FileResponse(os.path.join(BASE_DIR, "booth.html"))


@app.get("/listen")
async def listen_page():
    return FileResponse(os.path.join(BASE_DIR, "listen.html"))


# ─── Listener push helper ─────────────────────────────────────────────────────

async def push_audio_to_listeners(lang: str, audio_bytes: bytes) -> None:
    """Send MP3 bytes to every listener subscribed to `lang`."""
    async with listeners_lock:
        sockets = set(listeners.get(lang, set()))

    stale: Set[WebSocket] = set()
    for ws in sockets:
        try:
            await ws.send_bytes(audio_bytes)
        except Exception:
            stale.add(ws)

    if stale:
        async with listeners_lock:
            listeners[lang] -= stale
        logger.info(f"Pruned {len(stale)} stale [{lang}] listener(s).")


# ─── WebSocket: Booth ─────────────────────────────────────────────────────────

@app.websocket("/ws/booth")
async def ws_booth(websocket: WebSocket) -> None:
    """
    Accepts the translation-booth browser connection.

    The booth streams raw WebM/Opus binary frames produced by MediaRecorder
    (timeslice=4000, so one frame ≈ 4 seconds of audio).

    The first MediaRecorder frame contains the WebM container header (EBML +
    Tracks element).  Subsequent frames contain only Cluster data and are not
    independently decodable.  We cache the first frame and prepend it to every
    subsequent frame before forwarding to the Vast worker so that each
    forwarded chunk is a self-contained, decodable WebM stream.
    """
    global booth_ws
    await websocket.accept()
    booth_ws = websocket
    logger.info("Booth connected.")

    init_segment: Optional[bytes] = None

    try:
        while True:
            data: bytes = await websocket.receive_bytes()
            if not data:
                continue

            # Build a self-contained WebM chunk for the Vast worker.
            if init_segment is None:
                init_segment = data   # first frame: contains the WebM header
                chunk = data
            else:
                chunk = init_segment + data   # prepend header to make chunk decodable

            # Gather the languages that currently have at least one listener.
            async with listeners_lock:
                active_langs = [
                    lang for lang, sockets in listeners.items() if sockets
                ]

            if not active_langs:
                logger.debug("No active listeners – skipping chunk.")
                continue

            if vast_ws is None:
                logger.warning("Vast worker not connected – dropping chunk.")
                try:
                    await websocket.send_text(
                        json.dumps({"error": "Processing backend unavailable."})
                    )
                except Exception:
                    pass
                continue

            # Pack and forward the chunk to the Vast worker.
            chunk_id = str(uuid.uuid4())
            payload = pack_message(
                {"type": "process", "chunk_id": chunk_id, "active_langs": active_langs},
                chunk,
            )

            async with vast_send_lock:
                try:
                    await vast_ws.send(payload)
                except Exception as exc:
                    logger.error(f"Failed to send chunk to Vast: {exc}")

    except WebSocketDisconnect:
        logger.info("Booth disconnected.")
    except Exception as exc:
        logger.error(f"Booth WebSocket error: {exc}")
    finally:
        if booth_ws is websocket:
            booth_ws = None


# ─── WebSocket: Listener ──────────────────────────────────────────────────────

@app.websocket("/ws/listen/{lang}")
async def ws_listen(websocket: WebSocket, lang: str) -> None:
    """
    Accepts a browser listener connection for the given language channel.
    Audio is pushed to this socket whenever the Vast worker returns a result
    for `lang`.
    """
    if lang not in SUPPORTED_LANGUAGES:
        await websocket.close(code=4001, reason=f"Unsupported language: {lang!r}")
        return

    await websocket.accept()

    async with listeners_lock:
        listeners[lang].add(websocket)
        count = len(listeners[lang])

    logger.info(f"Listener joined [{lang}] (total for this lang: {count}).")

    try:
        # Listener is passive; keep the connection open until the client leaves.
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        logger.info(f"Listener disconnected [{lang}].")
    except Exception as exc:
        logger.error(f"Listener [{lang}] error: {exc}")
    finally:
        async with listeners_lock:
            listeners[lang].discard(websocket)
        logger.info(
            f"Listener removed [{lang}] "
            f"(remaining: {len(listeners[lang])})."
        )


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

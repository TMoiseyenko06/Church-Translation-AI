"""
Church Translation – VPS Relay Server
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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vps-relay")

VAST_WS_URL: str = os.environ.get("VAST_WS_URL", "ws://localhost:8888/ws/worker")
SUPPORTED_LANGUAGES: Dict[str, str] = {"en": "English"}

vast_ws: Optional[websockets.WebSocketClientProtocol] = None  # type: ignore
vast_send_lock = asyncio.Lock()
booth_ws: Optional[WebSocket] = None
listeners: Dict[str, Set[WebSocket]] = {lang: set() for lang in SUPPORTED_LANGUAGES}
listeners_lock = asyncio.Lock()


def pack_message(meta: dict, data: bytes) -> bytes:
    meta_bytes = json.dumps(meta).encode()
    return struct.pack(">I", len(meta_bytes)) + meta_bytes + data


def unpack_message(raw: bytes) -> tuple[dict, bytes]:
    if len(raw) < 4:
        raise ValueError("Frame too short.")
    json_len = struct.unpack(">I", raw[:4])[0]
    if len(raw) < 4 + json_len:
        raise ValueError("Frame truncated.")
    meta = json.loads(raw[4 : 4 + json_len])
    data = raw[4 + json_len :]
    return meta, data


# WebM Cluster EBML ID — marks start of actual audio data
_WEBM_CLUSTER_ID = bytes([0x1F, 0x43, 0xB6, 0x75])


def extract_webm_init(first_blob: bytes) -> tuple[bytes, bytes]:
    """Split the first MediaRecorder blob into (pure_headers, first_cluster_data).

    The first blob from MediaRecorder contains the WebM init segment (EBML
    header + Segment info + Tracks) followed immediately by the first audio
    Cluster.  We only want the header portion as the init segment so that we
    don't re-play the first chunk's audio on every subsequent chunk.
    """
    idx = first_blob.find(_WEBM_CLUSTER_ID)
    if idx == -1:
        return first_blob, b""
    return first_blob[:idx], first_blob[idx:]


async def vast_receive_loop(ws) -> None:
    global booth_ws
    async for raw_msg in ws:
        if not isinstance(raw_msg, bytes):
            continue
        try:
            meta, audio_bytes = unpack_message(raw_msg)
        except Exception as exc:
            logger.error(f"Failed to unpack Vast frame: {exc}")
            continue

        if meta.get("type") == "result":
            lang       = meta.get("lang", "")
            transcript = meta.get("transcript", "")
            detected   = meta.get("detected_lang", "")

            if transcript and booth_ws:
                try:
                    await booth_ws.send_text(
                        json.dumps({"transcript": transcript, "language": detected})
                    )
                except Exception:
                    pass

            if lang and audio_bytes:
                await push_audio_to_listeners(lang, audio_bytes)

        elif meta.get("type") == "error":
            logger.error(f"Vast worker error: {meta.get('message')}")


async def connect_to_vast_loop() -> None:
    global vast_ws
    backoff = 2
    while True:
        try:
            logger.info(f"Connecting to Vast worker at {VAST_WS_URL} …")
            async with websockets.connect(
                VAST_WS_URL,
                ping_interval=20,
                ping_timeout=30,
                max_size=100 * 1024 * 1024,
            ) as ws:
                vast_ws = ws
                backoff = 2
                logger.info("Connected to Vast worker.")
                await vast_receive_loop(ws)
        except websockets.exceptions.ConnectionClosedOK:
            logger.info("Vast worker closed connection cleanly.")
        except Exception as exc:
            logger.error(f"Vast connection error: {exc}")
        finally:
            vast_ws = None
        logger.info(f"Reconnecting in {backoff}s …")
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(connect_to_vast_loop())
    logger.info("VPS relay starting.")
    yield
    task.cancel()
    logger.info("VPS relay stopped.")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _read_html(name: str) -> str:
    with open(os.path.join(BASE_DIR, name), encoding="utf-8") as f:
        return f.read()


@app.get("/booth", response_class=HTMLResponse)
async def booth_page():
    return HTMLResponse(content=_read_html("booth.html"))


@app.get("/listen", response_class=HTMLResponse)
async def listen_page():
    return HTMLResponse(content=_read_html("listen.html"))


@app.get("/health")
async def health():
    return {"status": "ok", "vast_connected": vast_ws is not None}


async def push_audio_to_listeners(lang: str, audio_bytes: bytes) -> None:
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


@app.websocket("/ws/booth")
async def ws_booth(websocket: WebSocket) -> None:
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

            if init_segment is None:
                # Extract pure WebM headers (no audio) from the first blob
                init_segment, first_cluster = extract_webm_init(data)
                if not first_cluster:
                    # No audio in first blob yet — wait for next
                    continue
                chunk = init_segment + first_cluster  # equivalent to original data
            else:
                chunk = init_segment + data

            if vast_ws is None:
                try:
                    await websocket.send_text(json.dumps({"error": "Backend unavailable."}))
                except Exception:
                    pass
                continue

            async with listeners_lock:
                active_langs = [l for l, s in listeners.items() if s] or list(SUPPORTED_LANGUAGES.keys())

            chunk_id = str(uuid.uuid4())
            payload = pack_message(
                {"type": "process", "chunk_id": chunk_id, "active_langs": active_langs},
                chunk,
            )
            async with vast_send_lock:
                try:
                    await vast_ws.send(payload)
                except Exception as exc:
                    logger.error(f"Failed to send to Vast: {exc}")

    except WebSocketDisconnect:
        logger.info("Booth disconnected.")
    except Exception as exc:
        logger.error(f"Booth error: {exc}")
    finally:
        if booth_ws is websocket:
            booth_ws = None


@app.websocket("/ws/listen/{lang}")
async def ws_listen(websocket: WebSocket, lang: str) -> None:
    if lang not in SUPPORTED_LANGUAGES:
        await websocket.close(code=4001, reason=f"Unsupported language: {lang!r}")
        return
    await websocket.accept()
    async with listeners_lock:
        listeners[lang].add(websocket)
    logger.info(f"Listener joined [{lang}].")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        async with listeners_lock:
            listeners[lang].discard(websocket)
        logger.info(f"Listener left [{lang}].")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

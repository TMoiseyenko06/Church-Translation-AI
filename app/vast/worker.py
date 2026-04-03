"""
Church Translation – Vast AI Worker
=====================================
Responsibilities:
  • Accept one persistent WebSocket connection from the VPS relay
  • For each audio chunk received:
      1. Decode the framed message (JSON header + WebM bytes)
      2. Convert WebM → 16 kHz mono WAV (ffmpeg subprocess)
      3. Transcribe with faster-whisper (large-v3, GPU)
      4. Detect source language from Whisper metadata
      5. For each language in active_langs (sent by VPS):
            a. Translate with deep-translator (Google, free tier)
            b. Synthesise speech with edge-tts (Microsoft neural voices)
            c. Pack a result frame and send it back to the VPS relay
  • Incoming chunks are queued so the receive loop stays responsive even
    when pipeline processing takes longer than the chunk interval.

NO browser WebSocket connections are made here – all client comms go
through the VPS relay.

Framing protocol (bidirectional, binary WebSocket messages):
  ┌──────────────────────────────────────────────────────┐
  │  4 bytes (big-endian uint32) = length of JSON header │
  │  N bytes  = UTF-8 JSON header                        │
  │  M bytes  = binary payload (WebM audio / MP3 audio)  │
  └──────────────────────────────────────────────────────┘

VPS → Vast  header: {"type":"process", "chunk_id":"<uuid>", "active_langs":["en","es"]}
Vast → VPS  header: {"type":"result",  "chunk_id":"<uuid>", "lang":"en",
                      "transcript":"…", "detected_lang":"en"}
             (transcript is populated only on the first result per chunk)
"""

import asyncio
import json
import logging
import os
import struct
import subprocess
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

import edge_tts
import uvicorn
from deep_translator import GoogleTranslator
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vast-worker")

# ─── Language / voice configuration ──────────────────────────────────────────

VOICE_MAP: dict[str, str] = {
    "en": "en-US-AriaNeural",
    "es": "es-ES-ElviraNeural",
    "fr": "fr-FR-DeniseNeural",
    "pt": "pt-BR-FranciscaNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "ko": "ko-KR-SunHiNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "ar": "ar-SA-ZariyahNeural",
}

# ─── Global state ─────────────────────────────────────────────────────────────

whisper_model: Optional[WhisperModel] = None

# ─── App lifecycle ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global whisper_model
    logger.info("Loading faster-whisper large-v3 on GPU …")
    whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    logger.info("Whisper model ready.")
    yield
    logger.info("Worker shutting down.")


app = FastAPI(title="Church Translation Vast Worker", lifespan=lifespan)

# ─── Framing helpers ──────────────────────────────────────────────────────────

def pack_message(meta: dict, data: bytes) -> bytes:
    """Encode (metadata-dict, binary-payload) into a single framed binary message."""
    meta_bytes = json.dumps(meta).encode()
    return struct.pack(">I", len(meta_bytes)) + meta_bytes + data


def unpack_message(raw: bytes) -> tuple[dict, bytes]:
    """Decode a raw binary frame into (metadata-dict, binary-payload)."""
    if len(raw) < 4:
        raise ValueError("Frame too short.")
    json_len = struct.unpack(">I", raw[:4])[0]
    if len(raw) < 4 + json_len:
        raise ValueError("Frame JSON region truncated.")
    meta = json.loads(raw[4 : 4 + json_len])
    data = raw[4 + json_len :]
    return meta, data


# ─── Stage 1 – WebM → WAV conversion (blocking, runs in thread executor) ──────

def convert_webm_to_wav(webm_bytes: bytes) -> str:
    """
    Write WebM bytes to a temp file, run ffmpeg to produce a 16 kHz mono WAV.
    Returns the WAV file path; caller must delete it.
    """
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(webm_bytes)
        webm_path = f.name

    wav_path = webm_path.replace(".webm", ".wav")

    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", webm_path,
            "-ar", "16000",   # Whisper expects 16 kHz
            "-ac", "1",       # mono
            "-f", "wav",
            wav_path,
        ],
        capture_output=True,
        timeout=60,
    )
    os.unlink(webm_path)

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (rc={result.returncode}): "
            f"{result.stderr.decode(errors='replace')}"
        )
    return wav_path


# ─── Stage 2 – Transcription (blocking, runs in thread executor) ──────────────

def transcribe_audio(wav_path: str) -> tuple[str, str]:
    """
    Run faster-whisper on the WAV file.
    Returns (full_transcript_text, detected_language_code).
    VAD filter suppresses silent segments before inference.
    """
    segments, info = whisper_model.transcribe(  # type: ignore[union-attr]
        wav_path,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text, info.language


# ─── Stage 3 – Translation (async, HTTP call offloaded to thread) ─────────────

async def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translate using deep-translator's GoogleTranslator (free, no API key).
    The blocking HTTP call is run in a thread executor.
    """
    if source_lang == target_lang:
        return text

    translator = GoogleTranslator(source=source_lang, target=target_lang)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, translator.translate, text)


# ─── Stage 4 – TTS synthesis (async, edge-tts native streaming) ───────────────

async def synthesize_speech(text: str, lang: str) -> bytes:
    """
    Synthesise `text` with the Microsoft neural voice for `lang`.
    Streams audio chunks from edge-tts and concatenates them into one MP3 blob.
    """
    voice = VOICE_MAP[lang]
    communicate = edge_tts.Communicate(text, voice)

    audio = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio.extend(chunk["data"])

    return bytes(audio)


# ─── Pipeline orchestration ───────────────────────────────────────────────────

async def process_chunk(
    websocket: WebSocket,
    chunk_id: str,
    active_langs: list[str],
    webm_bytes: bytes,
    send_lock: asyncio.Lock,
) -> None:
    """
    Full pipeline for one audio chunk.  Runs translation + TTS for all
    active languages concurrently after transcription completes.

    Results are sent back to the VPS relay as framed binary messages.
    """
    loop = asyncio.get_event_loop()

    # ── Stage 1: convert WebM → WAV ──────────────────────────────────────────
    try:
        wav_path = await loop.run_in_executor(None, convert_webm_to_wav, webm_bytes)
    except Exception as exc:
        logger.error(f"[{chunk_id}] ffmpeg error: {exc}")
        await _send_error(websocket, send_lock, chunk_id, str(exc))
        return

    try:
        # ── Stage 2: transcribe ───────────────────────────────────────────────
        try:
            text, detected_lang = await loop.run_in_executor(
                None, transcribe_audio, wav_path
            )
        except Exception as exc:
            logger.error(f"[{chunk_id}] Whisper error: {exc}")
            await _send_error(websocket, send_lock, chunk_id, str(exc))
            return

        logger.info(f"[{chunk_id}] Transcript [{detected_lang}]: {text!r}")

        if not text:
            logger.info(f"[{chunk_id}] Empty transcript (silence) – skipping.")
            return

        # ── Stages 3 & 4: translate + synthesise for each language, concurrently
        tasks = [
            _process_one_language(
                websocket, send_lock,
                chunk_id, text, detected_lang, lang,
                include_transcript=(i == 0),   # only the first result carries the transcript
            )
            for i, lang in enumerate(active_langs)
            if lang in VOICE_MAP
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


async def _process_one_language(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    chunk_id: str,
    text: str,
    source_lang: str,
    target_lang: str,
    include_transcript: bool,
) -> None:
    """Translate → synthesise → send result for a single target language."""
    try:
        # Stage 3: translate
        translated = await translate_text(text, source_lang, target_lang)
        logger.info(f"[{chunk_id}] [{target_lang}] → {translated!r}")

        # Stage 4: synthesise TTS
        audio_bytes = await synthesize_speech(translated, target_lang)

        # Pack and send result frame to VPS
        meta = {
            "type":          "result",
            "chunk_id":      chunk_id,
            "lang":          target_lang,
            "detected_lang": source_lang,
            # Include the transcript only once (first language result) so the
            # VPS can echo it to the booth without duplicating the message.
            "transcript":    text if include_transcript else "",
        }
        frame = pack_message(meta, audio_bytes)

        async with send_lock:
            await websocket.send_bytes(frame)

        logger.info(
            f"[{chunk_id}] [{target_lang}] sent {len(audio_bytes):,} bytes of audio."
        )

    except Exception as exc:
        logger.error(f"[{chunk_id}] [{target_lang}] processing error: {exc}")


async def _send_error(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    chunk_id: str,
    message: str,
) -> None:
    """Send an error notification frame back to the VPS relay."""
    frame = pack_message(
        {"type": "error", "chunk_id": chunk_id, "message": message},
        b"",
    )
    try:
        async with send_lock:
            await websocket.send_bytes(frame)
    except Exception:
        pass


# ─── WebSocket: VPS relay connection ─────────────────────────────────────────

@app.websocket("/ws/worker")
async def ws_worker(websocket: WebSocket) -> None:
    """
    Accepts one persistent WebSocket connection from the VPS relay.

    A dedicated asyncio Queue serialises incoming chunks so the receive
    loop stays unblocked even when pipeline processing takes > 4 seconds.
    A single processing worker task drains the queue.
    """
    await websocket.accept()
    logger.info(f"VPS relay connected from {websocket.client}.")

    # Mutex protecting concurrent sends back to the VPS.
    send_lock = asyncio.Lock()

    # Queue of (chunk_id, active_langs, webm_bytes) tuples.
    queue: asyncio.Queue[tuple[str, list[str], bytes]] = asyncio.Queue()

    async def processing_worker() -> None:
        """Drain the queue, processing one chunk at a time."""
        while True:
            chunk_id, active_langs, webm_bytes = await queue.get()
            try:
                await process_chunk(
                    websocket, chunk_id, active_langs, webm_bytes, send_lock
                )
            except Exception as exc:
                logger.error(f"Unhandled error in processing_worker: {exc}")
            finally:
                queue.task_done()

    worker_task = asyncio.create_task(processing_worker())

    try:
        while True:
            # Receive a framed binary message from the VPS relay.
            raw: bytes = await websocket.receive_bytes()

            try:
                meta, webm_bytes = unpack_message(raw)
            except Exception as exc:
                logger.error(f"Failed to unpack VPS frame: {exc}")
                continue

            if meta.get("type") != "process":
                logger.warning(f"Unexpected message type: {meta.get('type')}")
                continue

            chunk_id    = meta.get("chunk_id", "unknown")
            active_langs: list[str] = meta.get("active_langs", [])

            logger.info(
                f"Queued chunk [{chunk_id}] "
                f"for langs={active_langs}, "
                f"size={len(webm_bytes):,} bytes."
            )
            await queue.put((chunk_id, active_langs, webm_bytes))

    except WebSocketDisconnect:
        logger.info("VPS relay disconnected.")
    except Exception as exc:
        logger.error(f"Worker WebSocket error: {exc}")
    finally:
        worker_task.cancel()
        logger.info("Processing worker stopped.")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("worker:app", host="0.0.0.0", port=8001, reload=False)

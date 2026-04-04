"""
Church Translation – Vast AI Worker
=====================================
Pipeline per audio chunk (Russian sermon → English):
  1. Receive framed binary message from VPS relay (JSON header + WebM bytes)
  2. Convert WebM → 16 kHz mono WAV  (ffmpeg subprocess)
  3. Transcribe with faster-whisper large-v3-turbo, language auto-detected
  4. If detected language is already English → pass through unchanged
     Otherwise → translate Russian → English via NLLB-200-distilled-1.3B
     (Facebook's dedicated neural translation model, runs on GPU)
  5. Synthesise English speech with edge-tts (Microsoft neural male voice)
  6. Send resulting MP3 bytes back to VPS relay

Framing protocol (shared with VPS relay, binary WebSocket messages):
  [4-byte big-endian uint32 = JSON length][JSON bytes][audio bytes]

VPS → Vast  header: {"type":"process","chunk_id":"<uuid>","active_langs":["en"]}
Vast → VPS  header: {"type":"result","chunk_id":"<uuid>","lang":"en",
                      "transcript":"...","detected_lang":"ru"}
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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
from transformers import pipeline as hf_pipeline

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vast-worker")

# ─── Configuration ────────────────────────────────────────────────────────────

# Microsoft neural male voice for English output.
# Alternatives: en-US-GuyNeural, en-US-EricNeural, en-GB-RyanNeural
TTS_VOICE: str = "en-US-ChristopherNeural"
TTS_RATE:  str = "+50%"

# Minimum characters in transcript before attempting translation + TTS.
MIN_TRANSCRIPT_CHARS: int = 4

# Minimum seconds of real speech (after VAD) to process a chunk.
# Prevents Whisper hallucinations on near-silence.
MIN_SPEECH_SECONDS: float = 1.0

# NLLB language codes
NLLB_SRC_LANG = "rus_Cyrl"   # Russian
NLLB_TGT_LANG = "eng_Latn"   # English

# ─── Global state ─────────────────────────────────────────────────────────────

whisper_model: Optional[WhisperModel] = None
translator = None  # HuggingFace translation pipeline

# ─── Application lifecycle ────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global whisper_model, translator

    logger.info("Loading faster-whisper large-v3-turbo on GPU …")
    whisper_model = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8_float16")
    logger.info("Whisper ready.")

    logger.info("Loading NLLB-200-distilled-1.3B translation model on GPU …")
    translator = hf_pipeline(
        "translation",
        model="facebook/nllb-200-distilled-1.3B",
        device=0,  # GPU 0
        src_lang=NLLB_SRC_LANG,
        tgt_lang=NLLB_TGT_LANG,
        max_length=512,
    )
    logger.info("Translation model ready.")
    logger.info(f"TTS voice: {TTS_VOICE} at {TTS_RATE}")
    yield
    logger.info("Worker shutting down.")


app = FastAPI(title="Church Translation Vast Worker", lifespan=lifespan)

# ─── Framing helpers ──────────────────────────────────────────────────────────

def pack_message(meta: dict, data: bytes) -> bytes:
    meta_bytes = json.dumps(meta).encode()
    return struct.pack(">I", len(meta_bytes)) + meta_bytes + data


def unpack_message(raw: bytes) -> tuple[dict, bytes]:
    if len(raw) < 4:
        raise ValueError("Frame too short.")
    json_len = struct.unpack(">I", raw[:4])[0]
    if len(raw) < 4 + json_len:
        raise ValueError("Frame JSON region truncated.")
    meta = json.loads(raw[4 : 4 + json_len])
    data = raw[4 + json_len :]
    return meta, data


# ─── Stage 1 – WebM → WAV (blocking, thread executor) ────────────────────────

def convert_webm_to_wav(webm_bytes: bytes) -> str:
    """Save WebM bytes, ffmpeg-convert to 16 kHz mono WAV. Returns WAV path."""
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(webm_bytes)
        webm_path = f.name

    wav_path = webm_path.replace(".webm", ".wav")
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", webm_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
        capture_output=True, timeout=60,
    )
    os.unlink(webm_path)

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (rc={result.returncode}): "
            f"{result.stderr.decode(errors='replace')}"
        )
    return wav_path


# ─── Stage 2 – Transcription (blocking, thread executor) ─────────────────────

def transcribe_audio(wav_path: str) -> tuple[str, str]:
    """
    Transcribe with faster-whisper. Language detection always enabled.
    Returns (transcript_text, detected_language_code).
    Returns ("", language) if the chunk contains too little real speech.
    """
    segments, info = whisper_model.transcribe(  # type: ignore[union-attr]
        wav_path,
        beam_size=3,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    speech_seconds = getattr(info, "duration_after_vad", info.duration)
    if speech_seconds < MIN_SPEECH_SECONDS:
        logger.info(
            f"Skipping chunk — only {speech_seconds:.2f}s of speech after VAD "
            f"(threshold: {MIN_SPEECH_SECONDS}s)."
        )
        return "", info.language

    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text, info.language


# ─── Stage 3 – Neural translation (NLLB, blocking, thread executor) ──────────

def translate_sync(text: str, detected_lang: str) -> str:
    """
    Translate `text` to English using NLLB-200.
    Skips translation if the detected language is already English.
    Runs synchronously — call via run_in_executor.
    """
    if detected_lang == "en":
        logger.info("Detected English — skipping translation.")
        return text

    result = translator(  # type: ignore[operator]
        text,
        src_lang=NLLB_SRC_LANG,
        tgt_lang=NLLB_TGT_LANG,
        max_length=512,
    )
    return result[0]["translation_text"].strip()


# ─── Stage 4 – TTS synthesis (edge-tts, async) ───────────────────────────────

async def synthesize_speech(text: str) -> bytes:
    """
    Synthesise `text` with the configured Microsoft neural male voice.
    edge-tts streams MP3 chunks which are concatenated and returned.
    """
    communicate = edge_tts.Communicate(text, TTS_VOICE, rate=TTS_RATE)
    audio = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio.extend(chunk["data"])
    return bytes(audio)


# ─── Full pipeline orchestration ──────────────────────────────────────────────

async def process_chunk(
    websocket: WebSocket,
    chunk_id: str,
    webm_bytes: bytes,
    send_lock: asyncio.Lock,
) -> None:
    loop = asyncio.get_event_loop()

    # Stage 1: WebM → WAV
    try:
        wav_path = await loop.run_in_executor(None, convert_webm_to_wav, webm_bytes)
    except Exception as exc:
        logger.error(f"[{chunk_id}] ffmpeg error: {exc}")
        await _send_error(websocket, send_lock, chunk_id, str(exc))
        return

    try:
        # Stage 2: transcribe
        try:
            text, detected_lang = await loop.run_in_executor(
                None, transcribe_audio, wav_path
            )
        except Exception as exc:
            logger.error(f"[{chunk_id}] Whisper error: {exc}")
            await _send_error(websocket, send_lock, chunk_id, str(exc))
            return

        has_speech = len(text) >= MIN_TRANSCRIPT_CHARS
        logger.info(
            f"[{chunk_id}] Transcript [{detected_lang}]: {text!r} "
            f"(speech={'yes' if has_speech else 'no'})"
        )

        if not has_speech:
            return

        # Stage 3: translate
        try:
            translated = await loop.run_in_executor(
                None, translate_sync, text, detected_lang
            )
            logger.info(f"[{chunk_id}] Translated: {translated!r}")
        except Exception as exc:
            logger.error(f"[{chunk_id}] Translation error: {exc}")
            await _send_error(websocket, send_lock, chunk_id, str(exc))
            return

        # Stage 4: synthesise
        try:
            mp3_bytes = await synthesize_speech(translated)
        except Exception as exc:
            logger.error(f"[{chunk_id}] TTS error: {exc}")
            await _send_error(websocket, send_lock, chunk_id, str(exc))
            return

        # Send result frame to VPS
        frame = pack_message(
            {
                "type":          "result",
                "chunk_id":      chunk_id,
                "lang":          "en",
                "detected_lang": detected_lang,
                "transcript":    text,
            },
            mp3_bytes,
        )
        async with send_lock:
            await websocket.send_bytes(frame)

        logger.info(f"[{chunk_id}] Sent {len(mp3_bytes):,} bytes of MP3 to VPS.")

    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


async def _send_error(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    chunk_id: str,
    message: str,
) -> None:
    frame = pack_message({"type": "error", "chunk_id": chunk_id, "message": message}, b"")
    try:
        async with send_lock:
            await websocket.send_bytes(frame)
    except Exception:
        pass


# ─── WebSocket: VPS relay connection ─────────────────────────────────────────

@app.websocket("/ws/worker")
async def ws_worker(websocket: WebSocket) -> None:
    await websocket.accept()
    logger.info(f"VPS relay connected from {websocket.client}.")

    send_lock = asyncio.Lock()
    queue: asyncio.Queue[tuple[str, bytes]] = asyncio.Queue()

    async def pipeline_worker() -> None:
        while True:
            chunk_id, webm_bytes = await queue.get()
            # Drain backlog — skip stale chunks, only process the most recent.
            while not queue.empty():
                queue.task_done()
                chunk_id, webm_bytes = queue.get_nowait()
                logger.warning(f"Skipped stale chunk, processing latest [{chunk_id}].")
            try:
                await process_chunk(websocket, chunk_id, webm_bytes, send_lock)
            except Exception as exc:
                logger.error(f"Unhandled pipeline error: {exc}")
            finally:
                queue.task_done()

    worker_task = asyncio.create_task(pipeline_worker())

    try:
        while True:
            raw: bytes = await websocket.receive_bytes()
            try:
                meta, webm_bytes = unpack_message(raw)
            except Exception as exc:
                logger.error(f"Unpack error: {exc}")
                continue

            if meta.get("type") != "process":
                continue

            chunk_id     = meta.get("chunk_id", "unknown")
            active_langs = meta.get("active_langs", [])

            if "en" not in active_langs:
                continue

            logger.info(
                f"Queued chunk [{chunk_id}], size={len(webm_bytes):,} bytes, "
                f"queue depth={queue.qsize()}."
            )
            await queue.put((chunk_id, webm_bytes))

    except WebSocketDisconnect:
        logger.info("VPS relay disconnected.")
    except Exception as exc:
        logger.error(f"Worker WebSocket error: {exc}")
    finally:
        worker_task.cancel()
        logger.info("Pipeline worker stopped.")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("worker:app", host="0.0.0.0", port=8888, reload=False)

"""
Church Translation – Vast AI Worker
=====================================
Pipeline per audio chunk (Russian sermon → English):
  1. Receive framed binary message from VPS relay (JSON header + WebM bytes)
  2. Convert WebM → 16 kHz mono WAV  (ffmpeg subprocess)
  3. Transcribe with faster-whisper large-v3, language auto-detected
  4. If detected language is already English → skip translation (pass-through)
     Otherwise → translate Russian → English via local Ollama (Qwen2.5-14B),
     using a sermon-optimised system prompt
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
import httpx
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel

# Persistent async HTTP client reused across all translation requests.
# Created in lifespan so it shares the running event loop.
_ollama_client: Optional[httpx.AsyncClient] = None

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vast-worker")

# ─── Configuration ────────────────────────────────────────────────────────────

OLLAMA_URL: str   = os.environ.get("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "qwen2.5:32b")

# Microsoft neural male voice for English output.
# Alternatives: en-US-GuyNeural, en-US-EricNeural, en-GB-RyanNeural
TTS_VOICE: str = "en-US-ChristopherNeural"

# Minimum characters in transcript before attempting translation + TTS.
MIN_TRANSCRIPT_CHARS: int = 4

# Minimum seconds of real speech (after VAD) required to process a chunk.
# Whisper hallucinates common phrases ("Goodbye.", "Thanks.", etc.) when given
# near-silence; rejecting chunks with too little speech prevents this.
MIN_SPEECH_SECONDS: float = 1.0


# ─── Sermon translation prompt ────────────────────────────────────────────────

TRANSLATION_SYSTEM_PROMPT = """\
You are a live sermon interpreter. Translate Russian to English instantly.

STRICT OUTPUT RULE: reply with ONLY the translated sentence(s). \
No notes. No alternatives. No parentheses. No clarifications. \
No "Note:". No "Translation:". No extra lines. Just the translation.

Guidelines:
- Natural, fluent English. Pastoral tone.
- Preserve theological terms: благодать=grace, покаяние=repentance, \
  искупление=redemption, освящение=sanctification, благословение=blessing.
- Keep any English words that appear in the source unchanged.
- The input may be a sentence fragment — translate it as-is, nothing added.
"""

# ─── Global state ─────────────────────────────────────────────────────────────

whisper_model: Optional[WhisperModel] = None

# ─── Application lifecycle ────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global whisper_model, _ollama_client

    # large-v3-turbo: distilled Whisper, ~8× faster than large-v3 with
    # near-identical accuracy for Russian sermon speech.
    # int8_float16: quantised weights, faster GPU throughput than float16.
    logger.info("Loading faster-whisper large-v3-turbo on GPU …")
    whisper_model = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8_float16")
    logger.info("Whisper ready.")

    # Persistent connection pool — avoids TCP handshake overhead on every chunk.
    _ollama_client = httpx.AsyncClient(timeout=60.0)

    try:
        r = await _ollama_client.get(f"{OLLAMA_URL}/api/tags")
        r.raise_for_status()
        logger.info(f"Ollama reachable at {OLLAMA_URL} — model: {OLLAMA_MODEL}")
    except Exception as exc:
        logger.warning(
            f"Ollama not reachable at startup ({exc}). "
            "Translation will fail until Ollama is running."
        )

    logger.info(f"TTS voice: {TTS_VOICE}")
    yield

    await _ollama_client.aclose()
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
    Transcribe with faster-whisper. Language detection always enabled —
    the preacher may use English words mid-sentence.
    Returns (transcript_text, detected_language_code).
    Returns ("", language) if the chunk contains too little real speech.
    """
    segments, info = whisper_model.transcribe(  # type: ignore[union-attr]
        wav_path,
        beam_size=3,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    # Reject near-silence chunks to prevent Whisper hallucinations.
    speech_seconds = getattr(info, "duration_after_vad", info.duration)
    if speech_seconds < MIN_SPEECH_SECONDS:
        logger.info(
            f"Skipping chunk — only {speech_seconds:.2f}s of speech after VAD "
            f"(threshold: {MIN_SPEECH_SECONDS}s)."
        )
        return "", info.language

    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text, info.language


# ─── Stage 3 – LLM translation via Ollama ────────────────────────────────────

def _strip_model_notes(text: str) -> str:
    """
    Remove parenthetical notes, alternative suggestions, and 'Note:' lines
    that some models append despite being told not to.
    Keeps only lines that look like actual translation content.
    """
    import re
    lines = text.splitlines()
    clean = []
    for line in lines:
        stripped = line.strip()
        # Drop lines that are purely a note/comment block
        if re.match(r'^\(?(Note|Alternatively|Alternative|Comment|Clarification)\b', stripped, re.IGNORECASE):
            break  # everything after a note line is noise too
        clean.append(line)
    result = "\n".join(clean).strip()
    # Also strip trailing parenthetical that starts mid-text: "... (Note: ...)"
    result = re.sub(r'\s*\([^)]*[Nn]ote[^)]*\)\s*$', '', result).strip()
    return result or text  # fall back to original if we stripped everything


async def translate_with_llm(text: str, detected_lang: str) -> str:
    """
    Translate to English via Ollama. Skips the call if already English.
    """
    if detected_lang == "en":
        logger.info("Detected English — skipping translation.")
        return text

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
            {"role": "user",   "content": text},
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 200,
        },
    }

    response = await _ollama_client.post(f"{OLLAMA_URL}/api/chat", json=payload)  # type: ignore[union-attr]
    response.raise_for_status()
    raw = response.json()["message"]["content"].strip()
    return _strip_model_notes(raw)


# ─── Stage 4 – TTS synthesis (edge-tts, async) ───────────────────────────────

TTS_RATE: str = "+50%"


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
        logger.info(f"[{chunk_id}] Transcript [{detected_lang}]: {text!r} (speech={'yes' if has_speech else 'no'})")

        if not has_speech:
            return

        # Stage 3: translate
        try:
            translated = await translate_with_llm(text, detected_lang)
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

            chunk_id   = meta.get("chunk_id", "unknown")
            active_langs = meta.get("active_langs", [])

            if "en" not in active_langs:
                continue

            logger.info(f"Queued chunk [{chunk_id}], size={len(webm_bytes):,} bytes, queue depth={queue.qsize()}.")
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

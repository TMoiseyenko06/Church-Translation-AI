"""
Church Translation – Vast AI Worker
=====================================
Pipeline per audio chunk (Russian sermon → English):
  1. Receive framed binary message from VPS relay (JSON header + WebM bytes)
  2. Convert WebM → 16 kHz mono WAV  (ffmpeg subprocess)
  3. Transcribe with faster-whisper large-v3, language auto-detected.
     Whisper's output is already split into sentence-level segments.
  4. Translate each segment independently via local Ollama (Qwen2.5-14B).
     English segments are passed through without translation.
  5. Measure the preacher's speech rate (words/second from VAD duration)
     and compute a matching TTS rate so translated speech fits the same window.
  6. Synthesise each translated segment with edge-tts and send it to VPS
     as its own audio frame. The browser's gapless player chains them.

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

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vast-worker")

# ─── Configuration ────────────────────────────────────────────────────────────

OLLAMA_URL: str   = os.environ.get("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")

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
You are an expert simultaneous interpreter specialising in live Christian \
sermon translation from Russian to English.

Rules (follow precisely):
1. Translate the supplied Russian text into natural, fluent English.
2. Preserve theological and biblical terminology accurately \
   (e.g. "благодать" → "grace", "покаяние" → "repentance", \
   "искупление" → "redemption", "освящение" → "sanctification").
3. Maintain the speaker's rhetorical style and emotional register \
   (earnest, pastoral, authoritative — not flat or robotic).
4. If the text already contains English words or phrases, keep them unchanged.
5. The input may be a mid-sentence fragment from live audio — translate it \
   exactly as supplied, without adding explanatory context or padding.
6. Output ONLY the English translation. No commentary, no quotation marks, \
   no prefixes like "Translation:" — just the translated text.
"""

# ─── Global state ─────────────────────────────────────────────────────────────

whisper_model: Optional[WhisperModel] = None

# ─── Application lifecycle ────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global whisper_model

    logger.info("Loading faster-whisper large-v3 on GPU …")
    whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    logger.info("Whisper ready.")

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            r.raise_for_status()
        logger.info(f"Ollama reachable at {OLLAMA_URL} — model: {OLLAMA_MODEL}")
    except Exception as exc:
        logger.warning(
            f"Ollama not reachable at startup ({exc}). "
            "Translation will fail until Ollama is running."
        )

    logger.info(f"TTS voice: {TTS_VOICE}")
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

def transcribe_audio(wav_path: str) -> tuple[list[str], str, float]:
    """
    Transcribe with faster-whisper. Returns (segments, language, speech_seconds).

    `segments` is a list of sentence/phrase strings as Whisper naturally splits
    them — each will be translated and synthesised independently so listeners
    hear one sentence at a time rather than a joined blob of text.

    Returns an empty segments list if the chunk contains too little real speech
    (prevents Whisper hallucinations on near-silence chunks).
    """
    segments_iter, info = whisper_model.transcribe(  # type: ignore[union-attr]
        wav_path,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    speech_seconds = getattr(info, "duration_after_vad", info.duration)
    if speech_seconds < MIN_SPEECH_SECONDS:
        logger.info(
            f"Skipping chunk — only {speech_seconds:.2f}s of speech after VAD "
            f"(threshold: {MIN_SPEECH_SECONDS}s)."
        )
        return [], info.language, speech_seconds

    # Collect segments eagerly (the iterator is lazy; reading it here while
    # we still hold the executor thread keeps GPU usage sequential).
    segments = [seg.text.strip() for seg in segments_iter if seg.text.strip()]
    return segments, info.language, speech_seconds


# ─── Stage 3 – LLM translation via Ollama ────────────────────────────────────

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
            "num_predict": 512,
        },
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        response.raise_for_status()

    return response.json()["message"]["content"].strip()


# ─── Stage 4 – TTS rate matching ─────────────────────────────────────────────

# Approximate base speaking rate of Christopher Neural at rate="+0%".
# Measured empirically; adjust if the voice sounds consistently too fast/slow.
_TTS_BASE_WPS: float = 2.5  # words per second


def compute_tts_rate(translated_word_count: int, speech_seconds: float) -> str:
    """
    Return an edge-tts `rate` string that makes the synthesised speech fill
    roughly the same duration as the preacher's original utterance.

    translated_word_count: number of English words across all segments
    speech_seconds: VAD-filtered duration of the source audio (actual speech only)
    """
    if speech_seconds <= 0 or translated_word_count <= 0:
        return "+30%"  # safe fallback

    target_wps = translated_word_count / speech_seconds
    pct = (target_wps / _TTS_BASE_WPS - 1.0) * 100.0
    # Clamp: don't go slower than –10% (sounds drowsy) or faster than +80%
    pct = max(-10.0, min(80.0, pct))
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.0f}%"


# ─── Stage 4 – TTS synthesis (edge-tts, async) ───────────────────────────────

async def synthesize_speech(text: str, rate: str = "+30%") -> bytes:
    """
    Synthesise `text` with the configured Microsoft neural male voice at `rate`.
    edge-tts streams MP3 chunks which are concatenated and returned.
    """
    communicate = edge_tts.Communicate(text, TTS_VOICE, rate=rate)
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
    """
    Full pipeline for one audio chunk:
      WebM → WAV → Whisper segments → per-segment translation → rate-matched TTS
      → one audio frame sent to VPS per sentence.
    """
    loop = asyncio.get_event_loop()

    # Stage 1: WebM → WAV
    try:
        wav_path = await loop.run_in_executor(None, convert_webm_to_wav, webm_bytes)
    except Exception as exc:
        logger.error(f"[{chunk_id}] ffmpeg error: {exc}")
        await _send_error(websocket, send_lock, chunk_id, str(exc))
        return

    try:
        # Stage 2: transcribe → list of sentence segments
        try:
            segments, detected_lang, speech_secs = await loop.run_in_executor(
                None, transcribe_audio, wav_path
            )
        except Exception as exc:
            logger.error(f"[{chunk_id}] Whisper error: {exc}")
            await _send_error(websocket, send_lock, chunk_id, str(exc))
            return

        # Filter out empty/very-short segments (stray punctuation, etc.)
        segments = [s for s in segments if len(s) >= MIN_TRANSCRIPT_CHARS]

        if not segments:
            return

        logger.info(
            f"[{chunk_id}] [{detected_lang}] {len(segments)} segment(s) "
            f"in {speech_secs:.2f}s of speech."
        )

        # Stage 3: translate all segments up-front so we know the total word
        # count before computing the TTS rate.
        translations: list[str] = []
        for i, seg_text in enumerate(segments):
            logger.info(f"[{chunk_id}] Seg {i+1}/{len(segments)}: {seg_text!r}")
            try:
                translated = await translate_with_llm(seg_text, detected_lang)
            except Exception as exc:
                logger.error(f"[{chunk_id}] Translation error (seg {i+1}): {exc}")
                translated = seg_text  # fall back to original on error
            logger.info(f"[{chunk_id}] → {translated!r}")
            translations.append(translated)

        # Compute TTS rate from total translated words vs total speech duration.
        total_words = sum(len(t.split()) for t in translations)
        rate = compute_tts_rate(total_words, speech_secs)
        logger.info(
            f"[{chunk_id}] TTS rate: {rate}  "
            f"({total_words} words / {speech_secs:.2f}s speech)"
        )

        # Stage 4: synthesise + send each sentence individually.
        # The first segment carries the full transcript for the booth display.
        full_transcript = " ".join(segments)
        for i, (seg_text, translated) in enumerate(zip(segments, translations)):
            try:
                mp3_bytes = await synthesize_speech(translated, rate)
            except Exception as exc:
                logger.error(f"[{chunk_id}] TTS error (seg {i+1}): {exc}")
                continue

            frame = pack_message(
                {
                    "type":          "result",
                    "chunk_id":      chunk_id,
                    "lang":          "en",
                    "detected_lang": detected_lang,
                    # Send transcript only with the first segment to avoid
                    # the booth display updating repeatedly for the same chunk.
                    "transcript":    full_transcript if i == 0 else "",
                },
                mp3_bytes,
            )
            async with send_lock:
                await websocket.send_bytes(frame)

            logger.info(
                f"[{chunk_id}] Seg {i+1}/{len(segments)} sent "
                f"({len(mp3_bytes):,} bytes MP3)."
            )

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

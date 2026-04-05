"""
Church Translation – Vast AI Worker
=====================================
Pipeline per audio chunk (Russian sermon → English):
  1. Receive framed binary message from VPS relay (JSON header + WebM bytes)
  2. Convert WebM → 16 kHz mono WAV  (ffmpeg subprocess)
  3. Transcribe with faster-whisper large-v3-turbo, language auto-detected
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

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vast-worker")

# ─── Configuration ────────────────────────────────────────────────────────────

OLLAMA_URL: str   = os.environ.get("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

TTS_VOICE: str = "en-US-ChristopherNeural"
TTS_RATE:  str = "+30%"

MIN_TRANSCRIPT_CHARS: int = 4
MIN_SPEECH_SECONDS:   float = 1.0
MIN_LANG_PROBABILITY: float = 0.70  # skip chunk if language detection confidence is below this

# Known Whisper hallucination phrases for silence/noise — skip these verbatim
HALLUCINATION_PHRASES: set[str] = {
    "редактор субтитров а.егорова",
    "субтитры сделал дима",
    "субтитры создавались при поддержке фонда",
    "продолжение следует",
    "субтитры",
    "goodbye",
    "thank you for watching",
    "thanks for watching",
    "you",
}

# ─── Sermon translation prompt ────────────────────────────────────────────────

TRANSLATION_SYSTEM_PROMPT = """\
You are a live interpreter for a Russian evangelical Christian sermon. \
Translate each chunk of Russian speech into natural, flowing English.

STRICT OUTPUT RULE: reply with ONLY the English translation. \
No notes, no alternatives, no parentheses, no clarifications, no labels. \
Output must be English using only Latin characters — never Chinese, Arabic, \
Cyrillic, or any other script.

Biblical & theological language:
- Use KJV/ESV-style phrasing when quoting or paraphrasing Scripture \
  (e.g. "poured out like water", "my bones are out of joint", \
  "they pierced my hands and my feet").
- Standard terms: благодать=grace, покаяние=repentance, \
  искупление=redemption, освящение=sanctification, спасение=salvation, \
  благословение=blessing, Писание=Scripture, Евангелие=Gospel, \
  грех=sin, праведность=righteousness, вера=faith, молитва=prayer, \
  церковь=church, Дух Святой=Holy Spirit, Господь=Lord, Бог=God.
- Proper nouns: Голгофа=Calvary, Вифлеем=Bethlehem, Иерусалим=Jerusalem, \
  Мессия=Messiah, keep personal names transliterated (Пётр=Peter, \
  Иоанн=John, Мария=Mary, etc.).

Style:
- Pastoral, reverent, natural spoken English — not overly formal or wooden.
- Preserve the preacher's rhetorical emphasis and repetition.
- The input may be a mid-sentence fragment — translate it as-is, \
  do not add words to complete the thought.
- Keep any English words already in the source unchanged.
"""

# ─── Global state ─────────────────────────────────────────────────────────────

whisper_model: Optional[WhisperModel] = None
_ollama_client: Optional[httpx.AsyncClient] = None
_last_transcript: str = ""

# Rolling translation history for multi-turn context: list of (source, translation) pairs
_translation_history: list[tuple[str, str]] = []
TRANSLATION_HISTORY_SIZE: int = 3  # how many previous pairs to include as context

# ─── Application lifecycle ────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global whisper_model, _ollama_client

    logger.info("Loading faster-whisper large-v3-turbo on GPU …")
    whisper_model = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8_float16")
    logger.info("Whisper ready.")

    _ollama_client = httpx.AsyncClient(timeout=60.0)
    try:
        r = await _ollama_client.get(f"{OLLAMA_URL}/api/tags")
        r.raise_for_status()
        logger.info(f"Ollama reachable at {OLLAMA_URL} — model: {OLLAMA_MODEL}")
    except Exception as exc:
        logger.warning(f"Ollama not reachable at startup ({exc}).")

    logger.info(f"TTS voice: {TTS_VOICE} at {TTS_RATE}")
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


# ─── Stage 1 – WebM → WAV ────────────────────────────────────────────────────

def convert_webm_to_wav(webm_bytes: bytes) -> str:
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


# ─── Stage 2 – Transcription ─────────────────────────────────────────────────

def transcribe_audio(wav_path: str) -> tuple[str, str]:
    global _last_transcript

    segments, info = whisper_model.transcribe(  # type: ignore[union-attr]
        wav_path,
        beam_size=3,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,
        initial_prompt=_last_transcript or None,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
    )

    speech_seconds = getattr(info, "duration_after_vad", info.duration)
    if speech_seconds < MIN_SPEECH_SECONDS:
        logger.info(f"Skipping chunk — only {speech_seconds:.2f}s of speech after VAD.")
        return "", info.language

    if info.language_probability < MIN_LANG_PROBABILITY:
        logger.info(f"Skipping chunk — low language confidence ({info.language_probability:.2f}).")
        return "", info.language

    text = " ".join(seg.text.strip() for seg in segments).strip()

    if text.lower().strip(".,!?…") in HALLUCINATION_PHRASES:
        logger.info(f"Skipping chunk — known hallucination phrase: {text!r}")
        return "", info.language

    if text:
        _last_transcript = ((_last_transcript + " " + text)[-300:]).strip()
    return text, info.language


# ─── Stage 3 – Translation via Ollama ────────────────────────────────────────

def _strip_model_notes(text: str) -> str:
    import re
    lines = text.splitlines()
    clean = []
    for line in lines:
        if re.match(r'^\(?(Note|Alternatively|Alternative|Comment|Clarification)\b', line.strip(), re.IGNORECASE):
            break
        clean.append(line)
    result = "\n".join(clean).strip()
    result = re.sub(r'\s*\([^)]*[Nn]ote[^)]*\)\s*$', '', result).strip()
    return result or text


def _is_mostly_latin(text: str) -> bool:
    """Return True if >80% of letters in text are Latin (ASCII a-z/A-Z)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    latin = sum(1 for c in letters if ord(c) < 128)
    return (latin / len(letters)) >= 0.8


async def translate_with_llm(text: str, detected_lang: str) -> str:
    global _translation_history

    if detected_lang == "en":
        logger.info("Detected English — skipping translation.")
        return text

    # Build messages: system prompt + last N source→translation pairs as turns + current
    messages: list[dict] = [{"role": "system", "content": TRANSLATION_SYSTEM_PROMPT}]
    for src, tgt in _translation_history[-TRANSLATION_HISTORY_SIZE:]:
        messages.append({"role": "user",      "content": src})
        messages.append({"role": "assistant", "content": tgt})
    messages.append({"role": "user", "content": text})

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 200},
    }

    for attempt in range(2):
        response = await _ollama_client.post(f"{OLLAMA_URL}/api/chat", json=payload)  # type: ignore[union-attr]
        response.raise_for_status()
        result = _strip_model_notes(response.json()["message"]["content"].strip())
        if _is_mostly_latin(result):
            # Store this pair for future context
            _translation_history.append((text, result))
            if len(_translation_history) > TRANSLATION_HISTORY_SIZE + 2:
                _translation_history = _translation_history[-TRANSLATION_HISTORY_SIZE:]
            return result
        logger.warning(f"Translation attempt {attempt+1} returned non-Latin text, retrying.")
        payload["options"]["temperature"] = 0.1

    # Store even imperfect result so history stays continuous
    _translation_history.append((text, result))
    if len(_translation_history) > TRANSLATION_HISTORY_SIZE + 2:
        _translation_history = _translation_history[-TRANSLATION_HISTORY_SIZE:]
    return result


# ─── Stage 4 – TTS ───────────────────────────────────────────────────────────

async def synthesize_speech(text: str) -> bytes:
    communicate = edge_tts.Communicate(text, TTS_VOICE, rate=TTS_RATE)
    audio = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio.extend(chunk["data"])
    return bytes(audio)


# ─── Pipeline ────────────────────────────────────────────────────────────────

async def process_chunk(
    websocket: WebSocket,
    chunk_id: str,
    webm_bytes: bytes,
    send_lock: asyncio.Lock,
) -> None:
    loop = asyncio.get_event_loop()

    try:
        wav_path = await loop.run_in_executor(None, convert_webm_to_wav, webm_bytes)
    except Exception as exc:
        logger.error(f"[{chunk_id}] ffmpeg error: {exc}")
        return

    try:
        try:
            text, detected_lang = await loop.run_in_executor(None, transcribe_audio, wav_path)
        except Exception as exc:
            logger.error(f"[{chunk_id}] Whisper error: {exc}")
            return

        if len(text) < MIN_TRANSCRIPT_CHARS:
            logger.info(f"[{chunk_id}] No speech.")
            return

        logger.info(f"[{chunk_id}] Transcript [{detected_lang}]: {text!r}")

        try:
            translated = await translate_with_llm(text, detected_lang)
            logger.info(f"[{chunk_id}] Translated: {translated!r}")
        except Exception as exc:
            logger.error(f"[{chunk_id}] Translation error: {exc}")
            return

        try:
            mp3_bytes = await synthesize_speech(translated)
        except Exception as exc:
            logger.error(f"[{chunk_id}] TTS error: {exc}")
            return

        frame = pack_message(
            {"type": "result", "chunk_id": chunk_id, "lang": "en",
             "detected_lang": detected_lang, "transcript": text},
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


# ─── WebSocket ────────────────────────────────────────────────────────────────

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

            chunk_id     = meta.get("chunk_id", "unknown")
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

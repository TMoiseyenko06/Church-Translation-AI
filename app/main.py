"""
Church Live Translation Streaming – Backend
==========================================
Pipeline per audio chunk:
  1. Receive WebM/Opus bytes from booth MediaRecorder
  2. Convert to 16 kHz mono WAV via ffmpeg subprocess
  3. Transcribe with faster-whisper (large-v3, GPU)
  4. Detect source language from Whisper metadata
  5. For every active listener language channel:
       a. Translate text with deep-translator (Google, free)
       b. Synthesise speech with edge-tts (Microsoft neural voices)
       c. Push resulting MP3 bytes to all WebSocket listeners on that channel
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
from contextlib import asynccontextmanager
from typing import Dict, Set

import edge_tts
import uvicorn
from deep_translator import GoogleTranslator
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from faster_whisper import WhisperModel

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("church-translation")

# ─── Language / voice configuration ──────────────────────────────────────────

# Each key is the BCP-47 language code used throughout the system.
# "voice" is the edge-tts neural voice identifier.
SUPPORTED_LANGUAGES: Dict[str, dict] = {
    "en": {"name": "English",    "voice": "en-US-AriaNeural"},
    "es": {"name": "Spanish",    "voice": "es-ES-ElviraNeural"},
    "fr": {"name": "French",     "voice": "fr-FR-DeniseNeural"},
    "pt": {"name": "Portuguese", "voice": "pt-BR-FranciscaNeural"},
    "zh": {"name": "Chinese",    "voice": "zh-CN-XiaoxiaoNeural"},
    "ko": {"name": "Korean",     "voice": "ko-KR-SunHiNeural"},
    "ru": {"name": "Russian",    "voice": "ru-RU-SvetlanaNeural"},
    "ar": {"name": "Arabic",     "voice": "ar-SA-ZariyahNeural"},
}

# ─── Global mutable state ────────────────────────────────────────────────────

# Whisper model instance – loaded once at startup.
whisper_model: WhisperModel = None  # type: ignore[assignment]

# Active listener WebSockets, keyed by language code.
listeners: Dict[str, Set[WebSocket]] = {lang: set() for lang in SUPPORTED_LANGUAGES}

# Async lock protecting the listeners dict from concurrent mutations.
listeners_lock = asyncio.Lock()

# ─── Application lifecycle ───────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the Whisper model once before serving any requests."""
    global whisper_model
    logger.info("Loading faster-whisper large-v3 model (this may take a minute)…")
    # device="cuda" uses the GPU; compute_type="float16" halves VRAM usage.
    whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    logger.info("Whisper model ready.")
    yield
    logger.info("Server shutting down.")


app = FastAPI(title="Church Translation Streaming", lifespan=lifespan)

# ─── Static page routes ───────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.get("/booth")
async def booth_page():
    """Serve the audio-capture page for the translation booth."""
    return FileResponse(os.path.join(BASE_DIR, "booth.html"))


@app.get("/listen")
async def listen_page():
    """Serve the listener page for congregation members."""
    return FileResponse(os.path.join(BASE_DIR, "listen.html"))


# ─── Stage 1 – Audio conversion (blocking, runs in thread executor) ───────────

def convert_webm_to_wav(webm_bytes: bytes) -> str:
    """
    Write raw WebM/Opus bytes to a temp file, convert to 16 kHz mono WAV
    with ffmpeg, return the WAV file path.  Caller must delete the file.
    """
    # Write the incoming WebM bytes to a temporary file.
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(webm_bytes)
        webm_path = f.name

    wav_path = webm_path.replace(".webm", ".wav")

    result = subprocess.run(
        [
            "ffmpeg", "-y",          # overwrite output without prompting
            "-i", webm_path,
            "-ar", "16000",          # Whisper expects 16 kHz
            "-ac", "1",              # mono
            "-f", "wav",
            wav_path,
        ],
        capture_output=True,
        timeout=60,
    )

    os.unlink(webm_path)  # remove the temporary WebM file immediately

    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")
        raise RuntimeError(f"ffmpeg conversion failed: {stderr}")

    return wav_path


# ─── Stage 2 – Transcription (blocking, runs in thread executor) ──────────────

def transcribe_audio(wav_path: str) -> tuple[str, str]:
    """
    Run faster-whisper on a WAV file.
    Returns (transcribed_text, detected_language_code).
    VAD filter removes silent segments before transcription.
    """
    segments, info = whisper_model.transcribe(
        wav_path,
        beam_size=5,
        vad_filter=True,                              # skip silent regions
        vad_parameters={"min_silence_duration_ms": 500},
    )
    # Consume the generator and join all segment texts.
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text, info.language  # info.language is the two-letter code, e.g. "en"


# ─── Stage 3 – Translation (async, uses thread executor for HTTP call) ────────

async def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translate `text` from `source_lang` to `target_lang` using
    deep-translator's GoogleTranslator (free, no API key required).
    The underlying HTTP call is offloaded to a thread so it doesn't
    block the event loop.
    """
    if source_lang == target_lang:
        return text  # no-op if source and target are the same

    translator = GoogleTranslator(source=source_lang, target=target_lang)
    loop = asyncio.get_event_loop()
    translated: str = await loop.run_in_executor(None, translator.translate, text)
    return translated


# ─── Stage 4 – Text-to-Speech synthesis (async, native edge-tts streaming) ───

async def synthesize_speech(text: str, lang: str) -> bytes:
    """
    Use edge-tts to synthesise `text` with the neural voice for `lang`.
    Streams the audio chunks and concatenates them into a single MP3 blob.
    """
    voice = SUPPORTED_LANGUAGES[lang]["voice"]
    communicate = edge_tts.Communicate(text, voice)

    audio_data = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data.extend(chunk["data"])

    return bytes(audio_data)


# ─── Stage 5 – Push audio to listeners ───────────────────────────────────────

async def push_audio_to_listeners(lang: str, audio_bytes: bytes) -> None:
    """
    Send `audio_bytes` as a binary WebSocket message to every listener
    currently subscribed to `lang`.  Silently removes stale connections.
    """
    async with listeners_lock:
        # Snapshot the set so we can iterate outside the lock.
        sockets = set(listeners[lang])

    disconnected: Set[WebSocket] = set()
    for ws in sockets:
        try:
            await ws.send_bytes(audio_bytes)
        except Exception:
            # Treat any send failure as a disconnection.
            disconnected.add(ws)

    if disconnected:
        async with listeners_lock:
            listeners[lang] -= disconnected
        logger.info(f"Removed {len(disconnected)} stale [{lang}] listener(s).")


# ─── Per-language processing task ────────────────────────────────────────────

async def process_language_channel(
    text: str, source_lang: str, target_lang: str
) -> None:
    """
    Translate and synthesise audio for one language channel, then push it.
    Skips the channel if nobody is currently listening.
    """
    # Early-exit if no listeners are subscribed to this language.
    async with listeners_lock:
        if not listeners[target_lang]:
            return

    try:
        # Stage 3: translate
        translated = await translate_text(text, source_lang, target_lang)
        logger.info(f"  [{target_lang}] translated: {translated!r}")

        # Stage 4: synthesise
        audio_bytes = await synthesize_speech(translated, target_lang)

        # Stage 5: push to all listeners on this channel
        await push_audio_to_listeners(target_lang, audio_bytes)
        logger.info(f"  [{target_lang}] pushed {len(audio_bytes):,} bytes to listeners.")

    except Exception as exc:
        # A failure in one language channel must not affect others.
        logger.error(f"  [{target_lang}] processing error: {exc}")


# ─── Full pipeline orchestration ──────────────────────────────────────────────

async def process_audio_chunk(webm_bytes: bytes) -> tuple[str, str]:
    """
    Run the complete pipeline for one audio chunk:
      convert → transcribe → (translate + synthesise + push) × N languages

    Returns (transcript_text, detected_language) so the booth UI can display
    the live transcript.
    """
    loop = asyncio.get_event_loop()

    # Stage 1: convert WebM → WAV in a thread (blocking subprocess).
    wav_path = await loop.run_in_executor(None, convert_webm_to_wav, webm_bytes)

    try:
        # Stage 2: transcribe in a thread (blocking Whisper inference).
        text, detected_lang = await loop.run_in_executor(
            None, transcribe_audio, wav_path
        )
        logger.info(f"Transcript [{detected_lang}]: {text!r}")

        if not text:
            # Nothing to translate (silence or empty segment).
            return "", detected_lang

        # Stages 3-5: process all language channels concurrently.
        tasks = [
            process_language_channel(text, detected_lang, target_lang)
            for target_lang in SUPPORTED_LANGUAGES
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

        return text, detected_lang

    finally:
        # Always clean up the temporary WAV file.
        try:
            os.unlink(wav_path)
        except OSError:
            pass


# ─── WebSocket: Booth ─────────────────────────────────────────────────────────

@app.websocket("/ws/booth")
async def ws_booth(websocket: WebSocket) -> None:
    """
    Accepts a WebSocket connection from the translation booth.

    The booth sends raw WebM/Opus binary frames produced by the browser's
    MediaRecorder (with timeslice=4000).  The first frame contains the WebM
    container header (EBML + Tracks element); subsequent frames contain only
    Cluster data and are not independently decodable by ffmpeg.  We therefore
    prepend the saved init segment to every subsequent frame before processing.
    """
    await websocket.accept()
    logger.info("Booth connected.")

    # The first WebM blob from MediaRecorder includes the container header.
    # We cache it so subsequent chunks can be made self-contained for ffmpeg.
    init_segment: bytes | None = None

    try:
        while True:
            # Each message is a binary WebM chunk from MediaRecorder.ondataavailable.
            data: bytes = await websocket.receive_bytes()

            if not data:
                continue

            if init_segment is None:
                # First chunk: contains the WebM header AND the first cluster.
                # Save it verbatim as the init segment.
                init_segment = data
                chunk = data
            else:
                # Subsequent chunks: prepend the saved header so ffmpeg can
                # treat this as a valid, standalone WebM stream.
                chunk = init_segment + data

            try:
                text, detected_lang = await process_audio_chunk(chunk)
                if text:
                    # Echo the transcript back to the booth UI.
                    await websocket.send_text(
                        json.dumps({"transcript": text, "language": detected_lang})
                    )
            except Exception as exc:
                logger.error(f"Pipeline error on booth chunk: {exc}")
                # Inform the booth UI without dropping the connection.
                try:
                    await websocket.send_text(
                        json.dumps({"error": str(exc)})
                    )
                except Exception:
                    pass

    except WebSocketDisconnect:
        logger.info("Booth disconnected.")
    except Exception as exc:
        logger.error(f"Booth WebSocket fatal error: {exc}")


# ─── WebSocket: Listener ──────────────────────────────────────────────────────

@app.websocket("/ws/listen/{lang}")
async def ws_listen(websocket: WebSocket, lang: str) -> None:
    """
    Accepts a WebSocket connection from a congregation listener.

    The server pushes binary MP3 audio chunks whenever a translated segment
    for `lang` is ready.  The listener plays them back sequentially via the
    Web Audio API on the client side.
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
        # The listener is passive – it only receives audio.
        # We still need to read from the socket so the connection stays alive
        # and so we detect clean client-side closes (e.g. "Leave" button).
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
            f"Listener removed [{lang}] (remaining: {len(listeners[lang])})."
        )


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

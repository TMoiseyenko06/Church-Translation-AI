"""
Church Translation – Vast AI Worker
=====================================
Pipeline per audio chunk (Russian sermon → English):
  1. Receive framed binary message from VPS relay (JSON header + WebM bytes)
  2. Convert WebM → 16 kHz mono WAV  (ffmpeg subprocess)
  3. Transcribe with faster-whisper large-v3, language auto-detected
  4. If detected language is already English → skip translation (pass-through)
     Otherwise → translate Russian → English via local Ollama (Qwen2.5-7B),
     using a sermon-optimised system prompt
  5. Voice cloning: extract speaker embedding from incoming audio (XTTS v2),
     cache it, refresh every VOICE_REFRESH_INTERVAL chunks so the embedding
     tracks mic position changes over a long service
  6. Synthesise English speech in the preacher's cloned voice (XTTS v2 GPU)
  7. Convert synthesised WAV → MP3 (ffmpeg) and pack into a result frame
  8. Send result frame back to VPS relay

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

import httpx
import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel

# ── PyTorch 2.6+ compatibility patch ─────────────────────────────────────────
# PyTorch 2.6 changed torch.load to default weights_only=True, which breaks
# Coqui TTS checkpoint loading. Patch torch.load to restore the old default
# before TTS is imported so its internal calls succeed.
import torch as _torch
_orig_torch_load = _torch.load
def _patched_torch_load(f, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(f, *args, **kwargs)
_torch.load = _patched_torch_load
# ─────────────────────────────────────────────────────────────────────────────

# Accept Coqui TOS non-interactively (required before importing TTS).
# XTTS v2 is released under the Coqui Public Model License (non-commercial).
# Accept Coqui TOS non-interactively (required before importing TTS).
# XTTS v2 is released under the Coqui Public Model License (non-commercial).
os.environ["COQUI_TOS_AGREED"] = "1"
from TTS.api import TTS  # noqa: E402  (must come after env var and patch)

# ── XTTS audio loader patch ───────────────────────────────────────────────────
# Newer Coqui TTS versions use torchcodec for audio loading inside
# get_conditioning_latents(), but torchcodec is not available in all
# environments. Replace the loader with a torchaudio-based equivalent —
# torchaudio ships with PyTorch so it is always available.
import torchaudio as _torchaudio
import TTS.tts.layers.xtts.tokenizer as _xtts_tokenizer

def _load_audio_torchaudio(audiopath, sampling_rate):
    audio, sr = _torchaudio.load(audiopath)
    if sr != sampling_rate:
        audio = _torchaudio.transforms.Resample(sr, sampling_rate)(audio)
    if audio.shape[0] > 1:
        audio = audio.mean(0, keepdim=True)  # mix down to mono
    return audio.squeeze()

_xtts_tokenizer.load_audio = _load_audio_torchaudio
# ─────────────────────────────────────────────────────────────────────────────

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vast-worker")

# ─── Configuration ────────────────────────────────────────────────────────────

# Ollama endpoint and model for sermon translation.
OLLAMA_URL: str = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")

# How many processed chunks to wait between voice-embedding refreshes.
# At 4-second chunks, 8 chunks ≈ every ~32 seconds.
VOICE_REFRESH_INTERVAL: int = 8

# Minimum number of speech tokens Whisper must find before we attempt TTS.
# Protects against noisy/empty chunks producing garbled synthesis.
MIN_TRANSCRIPT_CHARS: int = 4

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
tts_model: Optional[TTS] = None  # Coqui XTTS v2

# Cached XTTS v2 voice conditioning latents extracted from the preacher's audio.
# Tuple of (gpt_cond_latent, speaker_embedding) tensors, or None until first capture.
voice_latents: Optional[tuple] = None
voice_lock = asyncio.Lock()          # protects voice_latents during update
voice_chunks_processed: int = 0      # counter used to schedule periodic refreshes

# ─── Application lifecycle ────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global whisper_model, tts_model

    # Load faster-whisper (GPU).
    logger.info("Loading faster-whisper large-v3 on GPU …")
    whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    logger.info("Whisper ready.")

    # Load XTTS v2 (GPU).  First run downloads ~1.9 GB of model weights to
    # ~/.local/share/tts/ — subsequent starts load from the local cache.
    logger.info("Loading XTTS v2 on GPU (may download ~1.9 GB on first run) …")
    tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    logger.info("XTTS v2 ready.")

    # Verify Ollama is reachable before accepting traffic.
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
    Transcribe with faster-whisper.  Language detection is always enabled —
    the preacher may switch to English mid-sentence.
    Returns (transcript_text, detected_language_code).
    """
    segments, info = whisper_model.transcribe(  # type: ignore[union-attr]
        wav_path,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        # No language= kwarg → auto-detect every chunk
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text, info.language


# ─── Stage 3 – LLM translation via Ollama ────────────────────────────────────

async def translate_with_llm(text: str, detected_lang: str) -> str:
    """
    Translate `text` to English using the local Ollama LLM.

    If Whisper already detected English (the preacher said a few English words),
    we skip the LLM call entirely and return the text as-is.
    """
    if detected_lang == "en":
        logger.info("Detected language is English — skipping translation.")
        return text

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
            {"role": "user",   "content": text},
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,   # low temperature → consistent, faithful translation
            "num_predict": 512,
        },
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        response.raise_for_status()

    content: str = response.json()["message"]["content"].strip()
    return content


# ─── Stage 4 – Voice embedding (blocking, thread executor) ───────────────────

def extract_voice_latents_sync(wav_path: str) -> tuple:
    """
    Extract XTTS v2 conditioning latents (speaker identity) from a WAV file.
    The reference audio should contain at least ~3 seconds of clean speech.
    Returns (gpt_cond_latent, speaker_embedding).
    """
    xtts = tts_model.synthesizer.tts_model  # type: ignore[union-attr]
    gpt_cond_latent, speaker_embedding = xtts.get_conditioning_latents(
        audio_path=[wav_path],
    )
    return gpt_cond_latent, speaker_embedding


async def maybe_update_voice(wav_path: str, has_speech: bool) -> None:
    """
    Update the cached voice latents from the current chunk's WAV.
    Only fires when:
      • We haven't captured a reference yet (first chunk with speech), OR
      • The periodic refresh interval has been reached.
    Runs the blocking extraction in a thread executor.
    """
    global voice_latents, voice_chunks_processed

    if not has_speech:
        return  # don't update from a silent chunk

    should_update = (voice_latents is None) or (
        voice_chunks_processed > 0
        and voice_chunks_processed % VOICE_REFRESH_INTERVAL == 0
    )

    if not should_update:
        return

    logger.info(f"Updating voice embedding (chunk #{voice_chunks_processed}) …")
    loop = asyncio.get_event_loop()
    try:
        latents = await loop.run_in_executor(
            None, extract_voice_latents_sync, wav_path
        )
        async with voice_lock:
            voice_latents = latents
        logger.info("Voice embedding updated.")
    except Exception as exc:
        logger.error(f"Voice extraction failed: {exc}")


# ─── Stage 5 – TTS synthesis (blocking, thread executor) ─────────────────────

def synthesize_cloned_speech_sync(
    text: str,
    gpt_cond_latent,
    speaker_embedding,
) -> np.ndarray:
    """
    Run XTTS v2 inference with the cached speaker conditioning.
    Returns a float32 numpy array at 24 000 Hz.
    """
    xtts = tts_model.synthesizer.tts_model  # type: ignore[union-attr]
    outputs = xtts.inference(
        text=text,
        language="en",
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=0.7,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=50,
        top_p=0.85,
        enable_text_splitting=True,   # handles long translated segments gracefully
    )
    return np.array(outputs["wav"], dtype=np.float32)


async def synthesize_speech(text: str) -> bytes:
    """
    Synthesise `text` in the preacher's cloned voice.
    Waits briefly if the voice embedding has not been captured yet
    (should only happen on the very first chunk).
    Returns MP3 bytes ready to stream to listeners.
    """
    # Wait up to 10 s for the first voice embedding to be captured.
    for _ in range(20):
        async with voice_lock:
            latents = voice_latents
        if latents is not None:
            break
        logger.info("Waiting for first voice embedding …")
        await asyncio.sleep(0.5)

    if latents is None:
        raise RuntimeError("No voice reference captured — cannot synthesise speech.")

    gpt_cond_latent, speaker_embedding = latents
    loop = asyncio.get_event_loop()

    # Run blocking XTTS inference in a thread.
    wav_array = await loop.run_in_executor(
        None, synthesize_cloned_speech_sync, text, gpt_cond_latent, speaker_embedding
    )

    # Convert numpy WAV array → MP3 bytes via ffmpeg.
    mp3_bytes = await loop.run_in_executor(None, wav_array_to_mp3, wav_array)
    return mp3_bytes


def wav_array_to_mp3(wav_array: np.ndarray, sample_rate: int = 24000) -> bytes:
    """Write numpy audio array to a temp WAV file, encode to MP3 with ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name

    sf.write(wav_path, wav_array, sample_rate)

    mp3_path = wav_path.replace(".wav", ".mp3")
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", wav_path, "-codec:a", "libmp3lame", "-q:a", "2", mp3_path],
        capture_output=True, timeout=60,
    )
    os.unlink(wav_path)

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg MP3 encode failed: {result.stderr.decode(errors='replace')}"
        )

    with open(mp3_path, "rb") as f:
        mp3_bytes = f.read()
    os.unlink(mp3_path)
    return mp3_bytes


# ─── Full pipeline orchestration ──────────────────────────────────────────────

async def process_chunk(
    websocket: WebSocket,
    chunk_id: str,
    webm_bytes: bytes,
    send_lock: asyncio.Lock,
) -> None:
    """
    Execute the full pipeline for one audio chunk and send the result
    frame back to the VPS relay.
    """
    global voice_chunks_processed
    loop = asyncio.get_event_loop()

    # ── Stage 1: convert WebM → WAV ──────────────────────────────────────────
    try:
        wav_path = await loop.run_in_executor(None, convert_webm_to_wav, webm_bytes)
    except Exception as exc:
        logger.error(f"[{chunk_id}] ffmpeg error: {exc}")
        await _send_error(websocket, send_lock, chunk_id, str(exc))
        return

    try:
        # ── Stage 2: transcribe + language detect ─────────────────────────────
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

        # Update voice embedding from this chunk (non-blocking background step).
        # Runs concurrently with translation so it doesn't add to the critical path.
        voice_update_task = asyncio.create_task(
            maybe_update_voice(wav_path, has_speech)
        )

        if not has_speech:
            await voice_update_task
            return

        voice_chunks_processed += 1

        # ── Stage 3: translate (or pass through if already English) ───────────
        try:
            translated = await translate_with_llm(text, detected_lang)
            logger.info(f"[{chunk_id}] Translated: {translated!r}")
        except Exception as exc:
            logger.error(f"[{chunk_id}] Translation error: {exc}")
            await _send_error(websocket, send_lock, chunk_id, str(exc))
            await voice_update_task
            return

        # Wait for voice embedding update to finish before synthesis
        # (only matters for the very first chunk).
        await voice_update_task

        # ── Stages 4-5: synthesise cloned speech ─────────────────────────────
        try:
            mp3_bytes = await synthesize_speech(translated)
        except Exception as exc:
            logger.error(f"[{chunk_id}] TTS error: {exc}")
            await _send_error(websocket, send_lock, chunk_id, str(exc))
            return

        # ── Send result frame to VPS relay ────────────────────────────────────
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

        logger.info(
            f"[{chunk_id}] Sent {len(mp3_bytes):,} bytes of MP3 audio to VPS."
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
    Accepts the single persistent WebSocket connection from the VPS relay.

    An asyncio Queue serialises incoming chunks: the receive loop puts chunks
    on the queue immediately so it never blocks, while a single worker task
    drains the queue one chunk at a time through the full pipeline.
    """
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
                logger.warning(f"Unexpected message type: {meta.get('type')!r}")
                continue

            chunk_id: str  = meta.get("chunk_id", "unknown")
            active_langs   = meta.get("active_langs", [])

            # This worker always produces English — only queue if "en" is active.
            if "en" not in active_langs:
                logger.debug(f"[{chunk_id}] No English listeners — skipping.")
                continue

            logger.info(
                f"Queued chunk [{chunk_id}], "
                f"size={len(webm_bytes):,} bytes, "
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

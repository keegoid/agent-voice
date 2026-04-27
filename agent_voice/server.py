"""FastAPI server for the agent-voice OpenAI-compatible audio subset."""

from __future__ import annotations

import io
import inspect
import json
import os
import re
import sys
import tempfile
import threading
import time
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .voices import VOICE_DESIGNS

TTS_MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
STT_MODEL_ID = "mlx-community/whisper-large-v3-mlx"
STT_PROCESSOR_ID = os.getenv("AGENT_VOICE_STT_PROCESSOR_ID", "openai/whisper-large-v3")
TTS_MAX_TOKENS = int(os.getenv("AGENT_VOICE_TTS_MAX_TOKENS") or os.getenv("CODEX_TTS_MAX_TOKENS") or "24000")
TTS_GENERATION_ATTEMPTS = int(
    os.getenv("AGENT_VOICE_TTS_GENERATION_ATTEMPTS") or os.getenv("CODEX_TTS_GENERATION_ATTEMPTS") or "2"
)
TTS_MAX_SEGMENT_CHARS = int(
    os.getenv("AGENT_VOICE_TTS_MAX_SEGMENT_CHARS") or os.getenv("CODEX_TTS_MAX_SEGMENT_CHARS") or "1200"
)
TTS_SEGMENT_SILENCE_SECONDS = float(
    os.getenv("AGENT_VOICE_TTS_SEGMENT_SILENCE_SECONDS")
    or os.getenv("CODEX_TTS_SEGMENT_SILENCE_SECONDS")
    or "0.18"
)
TTS_REQUEST_MAX_TOKENS_LIMIT = 100000
MAX_STT_UPLOAD_BYTES = int(
    os.getenv("AGENT_VOICE_MAX_STT_UPLOAD_BYTES")
    or os.getenv("CODEX_TTS_MAX_STT_UPLOAD_BYTES")
    or str(25 * 1024 * 1024)
)
ALLOWED_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
SAFE_STT_GENERATION_OPTIONS = {"language", "verbose", "max_tokens", "chunk_duration", "initial_prompt"}

app = FastAPI(title="agent-voice", version="0.2.1")
_tts_model = None
_stt_model = None
_tts_model_lock = threading.Lock()
_stt_model_lock = threading.Lock()
_dropped_stt_options_lock = threading.Lock()
_logged_dropped_stt_options: set[tuple[str, ...]] = set()


class RequestPayload(BaseModel):
    model: str = "qwen3-tts"
    input: str = Field(min_length=1)
    voice: str = "cyberpunk_cool"
    response_format: str = "wav"
    language: str = "English"
    instruct: str | None = Field(default=None, max_length=4000)
    max_tokens: int | None = Field(default=None, ge=1, le=TTS_REQUEST_MAX_TOKENS_LIMIT)


def get_tts_model():
    """Load the MLX model lazily; health checks must stay cheap."""
    global _tts_model
    if _tts_model is None:
        with _tts_model_lock:
            if _tts_model is not None:
                return _tts_model
            _tts_model = _load_tts_model()
    return _tts_model


def _load_tts_model():
    """Load the optional MLX runtime model."""
    if os.getenv("AGENT_VOICE_DISABLE_MODEL_LOAD") == "1" or os.getenv("CODEX_TTS_DISABLE_MODEL_LOAD") == "1":
        raise RuntimeError("model loading disabled")
    try:
        from mlx_audio.tts.utils import load_model as mlx_load_model
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError(
            "MLX audio runtime is not installed; run the installer or install the mlx extra"
        ) from exc

    started = time.perf_counter()
    print(f"Loading TTS model {TTS_MODEL_ID}...")
    model = mlx_load_model(TTS_MODEL_ID)
    print(f"TTS model loaded in {time.perf_counter() - started:.1f}s")
    return model


def get_stt_model():
    """Load the MLX Whisper model lazily; health checks must stay cheap."""
    global _stt_model
    if _stt_model is None:
        with _stt_model_lock:
            if _stt_model is not None:
                return _stt_model
            _stt_model = _load_stt_model()
    return _stt_model


def _load_stt_model():
    """Load the optional MLX runtime STT model."""
    if os.getenv("AGENT_VOICE_DISABLE_MODEL_LOAD") == "1" or os.getenv("CODEX_TTS_DISABLE_MODEL_LOAD") == "1":
        raise RuntimeError("model loading disabled")
    try:
        from mlx_audio.stt.utils import load_model as mlx_stt_load_model
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError(
            "MLX audio runtime is not installed; run the installer or install the mlx extra"
        ) from exc

    started = time.perf_counter()
    print(f"Loading STT model {STT_MODEL_ID}...")
    model = mlx_stt_load_model(STT_MODEL_ID)
    _ensure_stt_processor(model)
    print(f"STT model loaded in {time.perf_counter() - started:.1f}s")
    return model


def _ensure_stt_processor(model: Any) -> None:
    """Attach the Hugging Face Whisper processor when the MLX repo omits it."""
    if getattr(model, "_processor", None) is not None:
        return
    try:
        from transformers import WhisperProcessor
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError(
            "Transformers WhisperProcessor is not installed; run the installer or install the mlx extra"
        ) from exc
    try:
        model._processor = WhisperProcessor.from_pretrained(STT_PROCESSOR_ID)
    except Exception as exc:  # pragma: no cover - depends on network/cache state
        raise RuntimeError(f"Failed to load Whisper processor {STT_PROCESSOR_ID}") from exc


def generate_audio(
    text: str,
    instruct: str,
    language: str,
    response_format: str,
    max_tokens: int = TTS_MAX_TOKENS,
) -> bytes:
    """Generate audio bytes for a validated speech request."""
    model = get_tts_model()
    segments = _split_speech_text(text, TTS_MAX_SEGMENT_CHARS)
    chunks: list[np.ndarray] = []
    sample_rate = 24000
    output_sample_rate: int | None = None

    for index, segment in enumerate(segments):
        audio, sample_rate = _generate_audio_segment(
            model=model,
            text=segment,
            instruct=instruct,
            language=language,
            max_tokens=max_tokens,
        )
        if audio.size == 0:
            continue
        if output_sample_rate is None:
            output_sample_rate = sample_rate
        elif sample_rate != output_sample_rate:
            raise RuntimeError(
                f"TTS segments returned inconsistent sample rates: {output_sample_rate} and {sample_rate}"
            )
        if chunks and index > 0:
            chunks.append(np.zeros(int(sample_rate * TTS_SEGMENT_SILENCE_SECONDS), dtype=audio.dtype))
        chunks.append(audio)

    if not chunks:
        return b""

    audio = np.concatenate(chunks)
    output = io.BytesIO()
    sf.write(output, audio, output_sample_rate or sample_rate, format=response_format.upper())
    return output.getvalue()


def _generate_audio_segment(
    model: Any,
    text: str,
    instruct: str,
    language: str,
    max_tokens: int,
) -> tuple[np.ndarray, int]:
    attempts = max(1, TTS_GENERATION_ATTEMPTS)
    best_audio = np.array([], dtype=np.float32)
    best_sample_rate = 24000

    for attempt in range(attempts):
        audio, sample_rate, token_count = _generate_audio_segment_once(
            model=model,
            text=text,
            instruct=instruct,
            language=language,
            max_tokens=max_tokens,
            attempt=attempt,
        )
        if audio.size > best_audio.size:
            best_audio = audio
            best_sample_rate = sample_rate
        if not _is_suspiciously_short_audio(text, audio, sample_rate):
            return audio, sample_rate
        if attempt < attempts - 1:
            print(
                "Retrying suspiciously short TTS segment "
                f"(attempt={attempt + 1}, words={_word_count(text)}, seconds={audio.size / sample_rate:.2f}, "
                f"tokens={token_count})",
                file=sys.stderr,
            )

    return best_audio, best_sample_rate


def _generate_audio_segment_once(
    model: Any,
    text: str,
    instruct: str,
    language: str,
    max_tokens: int,
    attempt: int,
) -> tuple[np.ndarray, int, int]:
    chunks: list[np.ndarray] = []
    sample_rate = 24000
    token_count = 0
    temperature = 1.0 if attempt == 0 else 0.85
    top_p = 1.0 if attempt == 0 else 0.95
    repetition_penalty = 1.0 if attempt == 0 else 1.05

    for result in model.generate_voice_design(
        text=text,
        instruct=instruct,
        language=language,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
    ):
        chunks.append(np.asarray(result.audio))
        sample_rate = int(result.sample_rate)
        token_count += int(getattr(result, "token_count", 0) or 0)

    if not chunks:
        return np.array([], dtype=np.float32), sample_rate, token_count

    return np.concatenate(chunks), sample_rate, token_count


def _split_speech_text(text: str, max_segment_chars: int) -> list[str]:
    if max_segment_chars <= 0 or len(text) <= max_segment_chars:
        return [text]

    segments: list[str] = []
    current = ""
    for sentence in re.split(r"(?<=[.!?;:])\s+", text):
        if not sentence:
            continue
        if len(sentence) > max_segment_chars:
            if current:
                segments.append(current)
                current = ""
            segments.extend(_split_long_speech_piece(sentence, max_segment_chars))
            continue
        candidate = f"{current} {sentence}".strip()
        if len(candidate) > max_segment_chars and current:
            segments.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        segments.append(current)
    return segments or [text]


def _split_long_speech_piece(text: str, max_segment_chars: int) -> list[str]:
    segments: list[str] = []
    current = ""
    for word in text.split():
        if len(word) > max_segment_chars:
            if current:
                segments.append(current)
                current = ""
            segments.extend(word[start : start + max_segment_chars] for start in range(0, len(word), max_segment_chars))
            continue
        candidate = f"{current} {word}".strip()
        if len(candidate) > max_segment_chars and current:
            segments.append(current)
            current = word
        else:
            current = candidate
    if current:
        segments.append(current)
    return segments


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text))


def _is_suspiciously_short_audio(text: str, audio: np.ndarray, sample_rate: int) -> bool:
    words = _word_count(text)
    if words < 12 or sample_rate <= 0 or audio.size == 0:
        return False
    duration_seconds = audio.size / sample_rate
    # 6.5 words/sec is intentionally permissive; only retry obvious early-EOS clips.
    min_reasonable_seconds = max(2.0, words / 6.5)
    return duration_seconds < min_reasonable_seconds


def _sanitize_for_json(value: Any) -> Any:
    """Recursively replace non-JSON numeric values in model output."""
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, dict):
        return {key: _sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    return value


def _transcription_chunk_to_json(chunk: Any, accumulated: str) -> tuple[str, str]:
    if isinstance(chunk, str):
        accumulated += chunk
        return accumulated, json.dumps({"text": chunk, "accumulated": accumulated}) + "\n"
    if isinstance(chunk, dict):
        return accumulated, json.dumps(_sanitize_for_json(chunk)) + "\n"
    data = {
        "text": getattr(chunk, "text", ""),
        "start": getattr(chunk, "start_time", None),
        "end": getattr(chunk, "end_time", None),
        "is_final": getattr(chunk, "is_final", None),
        "language": getattr(chunk, "language", None),
    }
    return accumulated, json.dumps(_sanitize_for_json(data)) + "\n"


def _generate_transcription_lines(model: Any, tmp_path: str, gen_kwargs: dict[str, Any]) -> list[str]:
    """Generate newline-delimited transcription JSON, then remove the temp file."""
    try:
        result = model.generate(tmp_path, **gen_kwargs)
        if isinstance(result, str):
            return [json.dumps({"text": result}) + "\n"]
        if isinstance(result, dict):
            return [json.dumps(_sanitize_for_json(result)) + "\n"]
        if hasattr(result, "__iter__"):
            accumulated = ""
            lines = []
            for chunk in result:
                accumulated, payload = _transcription_chunk_to_json(chunk, accumulated)
                lines.append(payload)
            return lines
        return [json.dumps(_sanitize_for_json(result)) + "\n"]
    finally:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass


def _filter_generation_kwargs(model: Any, gen_kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(model.generate)
    except (TypeError, ValueError):
        filtered = {key: value for key, value in gen_kwargs.items() if key in SAFE_STT_GENERATION_OPTIONS}
        dropped = sorted(set(gen_kwargs) - set(filtered))
        if dropped:
            _log_dropped_stt_options_once(dropped)
        return filtered
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        allowed = set(signature.parameters) | SAFE_STT_GENERATION_OPTIONS
    else:
        allowed = set(signature.parameters)
    filtered = {key: value for key, value in gen_kwargs.items() if key in allowed}
    dropped = sorted(set(gen_kwargs) - set(filtered))
    if dropped:
        _log_dropped_stt_options_once(dropped)
    return filtered


def _log_dropped_stt_options_once(options: list[str]) -> None:
    key = tuple(options)
    with _dropped_stt_options_lock:
        if key in _logged_dropped_stt_options:
            return
        _logged_dropped_stt_options.add(key)
    print(f"Dropping unsupported STT generation options: {', '.join(options)}", file=sys.stderr)


async def _read_upload_limited(file: UploadFile, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(64 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(status_code=413, detail="file is too large")
        chunks.append(chunk)
    return b"".join(chunks)


@app.get("/v1/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model": "qwen3-tts",
        "tts_model_id": TTS_MODEL_ID,
        "tts_max_tokens": TTS_MAX_TOKENS,
        "stt_model_id": STT_MODEL_ID,
        "stt_processor_id": STT_PROCESSOR_ID,
        "voices": list(VOICE_DESIGNS.keys()),
    }


@app.post("/v1/audio/speech")
def audio_speech(request: RequestPayload) -> Response:
    if request.model not in {"qwen3-tts", TTS_MODEL_ID}:
        raise HTTPException(status_code=400, detail="Unsupported model")

    if request.response_format != "wav":
        raise HTTPException(status_code=400, detail="Unsupported response_format; only wav is supported")

    text = request.input.strip()
    if not text:
        raise HTTPException(status_code=400, detail="input must not be empty")

    custom_instruct = request.instruct.strip() if request.instruct else ""
    voice_design = custom_instruct or VOICE_DESIGNS.get(request.voice)
    if not voice_design:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Unknown voice '{request.voice}'",
                "voices": list(VOICE_DESIGNS.keys()),
            },
        )

    started = time.perf_counter()
    try:
        audio = generate_audio(
            text=text,
            instruct=voice_design,
            language=request.language,
            response_format=request.response_format,
            max_tokens=request.max_tokens or TTS_MAX_TOKENS,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Speech generation failed") from exc

    if not audio:
        raise HTTPException(status_code=500, detail="No audio generated")

    elapsed = time.perf_counter() - started
    return Response(
        content=audio,
        media_type="audio/wav",
        headers={"X-Generation-Time": f"{elapsed:.3f}"},
    )


@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    file: UploadFile = File(...),
    model: str = Form(STT_MODEL_ID),
    language: str | None = Form(None),
    verbose: bool = Form(False),
    max_tokens: int = Form(1024),
    chunk_duration: float = Form(30.0),
    frame_threshold: int | None = Form(None, include_in_schema=False),
    context: str | None = Form(None),
    prefill_step_size: int | None = Form(None, include_in_schema=False),
    text: str | None = Form(None),
) -> Response:
    """Transcribe audio with the MLX Whisper model as NDJSON."""
    if model != STT_MODEL_ID:
        raise HTTPException(status_code=400, detail="Unsupported model")

    data = await _read_upload_limited(file, MAX_STT_UPLOAD_BYTES)
    if not data:
        raise HTTPException(status_code=400, detail="file must not be empty")

    _, suffix = os.path.splitext(file.filename or "audio.wav")
    suffix = suffix.lower()
    if suffix not in ALLOWED_AUDIO_SUFFIXES:
        suffix = ".wav"
    with tempfile.NamedTemporaryFile(prefix="agent-voice-stt-", suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        stt_model = get_stt_model()
        gen_kwargs = {
            key: value
            for key, value in {
                "language": language,
                "verbose": verbose,
                "max_tokens": max_tokens,
                "chunk_duration": chunk_duration,
                "frame_threshold": frame_threshold,
                "context": context,
                "prefill_step_size": prefill_step_size,
                "text": text,
                "initial_prompt": context or text,
            }.items()
            if value is not None
        }
        gen_kwargs = _filter_generation_kwargs(stt_model, gen_kwargs)
        lines = _generate_transcription_lines(stt_model, tmp_path, gen_kwargs)
    except RuntimeError as exc:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass
        print(f"Transcription failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise HTTPException(status_code=500, detail="Transcription failed") from exc

    return Response(content="".join(lines), media_type="application/x-ndjson")


def main() -> None:
    # The service has no auth layer; keep the default host loopback-only.
    host, allow_remote = _server_bind_config()
    if host not in {"127.0.0.1", "localhost", "::1"} and not allow_remote:
        raise SystemExit("Refusing non-loopback AGENT_VOICE_HOST without AGENT_VOICE_ALLOW_REMOTE=1")
    port = int(os.getenv("AGENT_VOICE_PORT") or os.getenv("CODEX_TTS_PORT") or "8880")
    uvicorn.run("agent_voice.server:app", host=host, port=port)


def _server_bind_config() -> tuple[str, bool]:
    agent_host = os.getenv("AGENT_VOICE_HOST")
    if agent_host:
        return agent_host, os.getenv("AGENT_VOICE_ALLOW_REMOTE") == "1"
    return os.getenv("CODEX_TTS_HOST") or "127.0.0.1", os.getenv("CODEX_TTS_ALLOW_REMOTE") == "1"


if __name__ == "__main__":
    main()

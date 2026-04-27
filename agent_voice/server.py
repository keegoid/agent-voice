"""FastAPI server for the agent-voice OpenAI-compatible audio subset."""

from __future__ import annotations

import io
import inspect
import json
import os
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
MAX_STT_UPLOAD_BYTES = int(
    os.getenv("AGENT_VOICE_MAX_STT_UPLOAD_BYTES")
    or os.getenv("CODEX_TTS_MAX_STT_UPLOAD_BYTES")
    or str(25 * 1024 * 1024)
)
ALLOWED_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
SAFE_STT_GENERATION_OPTIONS = {"language", "verbose", "max_tokens", "chunk_duration", "initial_prompt"}

app = FastAPI(title="agent-voice", version="0.2.0")
_tts_model = None
_stt_model = None
_tts_model_lock = threading.Lock()
_stt_model_lock = threading.Lock()
_dropped_stt_options_lock = threading.Lock()
_logged_dropped_stt_options: set[tuple[str, ...]] = set()


class RequestPayload(BaseModel):
    model: str = "qwen3-tts"
    input: str = Field(min_length=1, max_length=4000)
    voice: str = "cyberpunk_cool"
    response_format: str = "wav"
    language: str = "English"
    instruct: str | None = Field(default=None, max_length=4000)


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
) -> bytes:
    """Generate audio bytes for a validated speech request."""
    model = get_tts_model()
    chunks: list[np.ndarray] = []
    sample_rate = 24000

    for result in model.generate_voice_design(
        text=text,
        instruct=instruct,
        language=language,
        temperature=1.0,
        top_p=1.0,
        repetition_penalty=1.0,
        max_tokens=12000,
    ):
        chunks.append(np.asarray(result.audio))
        sample_rate = int(result.sample_rate)

    if not chunks:
        return b""

    audio = np.concatenate(chunks)
    output = io.BytesIO()
    sf.write(output, audio, sample_rate, format=response_format.upper())
    return output.getvalue()


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
            print(f"Dropping unsupported STT generation options: {', '.join(dropped)}", file=sys.stderr)
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
    frame_threshold: int = Form(25),
    context: str | None = Form(None),
    prefill_step_size: int = Form(2048),
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

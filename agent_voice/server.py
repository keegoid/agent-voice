"""FastAPI server for the agent-voice OpenAI-compatible audio subset."""

from __future__ import annotations

import io
import inspect
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import warnings
import wave
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from . import mute_state
from .voices import VOICE_DESIGNS

TTS_MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
STT_MODEL_ID = "mlx-community/whisper-large-v3-mlx"
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
STT_PROCESSOR_ID = os.getenv("AGENT_VOICE_STT_PROCESSOR_ID", "openai/whisper-large-v3")
TTS_MAX_TOKENS = int(os.getenv("AGENT_VOICE_TTS_MAX_TOKENS") or "24000")
TTS_GENERATION_ATTEMPTS = int(os.getenv("AGENT_VOICE_TTS_GENERATION_ATTEMPTS") or "2")
TTS_MAX_SEGMENT_CHARS = int(os.getenv("AGENT_VOICE_TTS_MAX_SEGMENT_CHARS") or "1200")
TTS_SEGMENT_SILENCE_SECONDS = float(os.getenv("AGENT_VOICE_TTS_SEGMENT_SILENCE_SECONDS") or "0.18")
TTS_TEMPERATURE = float(os.getenv("AGENT_VOICE_TTS_TEMPERATURE") or "0.9")
TTS_TOP_P = float(os.getenv("AGENT_VOICE_TTS_TOP_P") or "0.95")
TTS_REPETITION_PENALTY = float(os.getenv("AGENT_VOICE_TTS_REPETITION_PENALTY") or "1.05")
TTS_RETRY_TEMPERATURE = float(os.getenv("AGENT_VOICE_TTS_RETRY_TEMPERATURE") or "0.75")
TTS_RETRY_TOP_P = float(os.getenv("AGENT_VOICE_TTS_RETRY_TOP_P") or "0.9")
TTS_RETRY_REPETITION_PENALTY = float(os.getenv("AGENT_VOICE_TTS_RETRY_REPETITION_PENALTY") or "1.1")
TTS_PEAK_LIMIT = float(os.getenv("AGENT_VOICE_TTS_PEAK_LIMIT") or "0.98")
TTS_SUSPICIOUS_MIN_WORDS = int(os.getenv("AGENT_VOICE_TTS_SUSPICIOUS_MIN_WORDS") or "4")
TTS_SUSPICIOUS_MAX_WORDS_PER_SECOND = float(
    os.getenv("AGENT_VOICE_TTS_SUSPICIOUS_MAX_WORDS_PER_SECOND") or "5.5"
)
TTS_SUSPICIOUS_MIN_SECONDS = float(os.getenv("AGENT_VOICE_TTS_SUSPICIOUS_MIN_SECONDS") or "1.0")
TTS_ACTIVITY_WINDOW_SECONDS = float(os.getenv("AGENT_VOICE_TTS_ACTIVITY_WINDOW_SECONDS") or "0.25")
TTS_ACTIVITY_MIN_RMS = float(os.getenv("AGENT_VOICE_TTS_ACTIVITY_MIN_RMS") or "0.004")
TTS_ACTIVITY_RELATIVE_RMS = float(os.getenv("AGENT_VOICE_TTS_ACTIVITY_RELATIVE_RMS") or "0.08")
TTS_ACTIVITY_MAX_SILENT_GAP_SECONDS = 0.75
TTS_REQUEST_MAX_TOKENS_LIMIT = 100000
FFMPEG_TIMEOUT_SECONDS = float(os.getenv("AGENT_VOICE_FFMPEG_TIMEOUT_SECONDS") or "60")
MAX_STT_UPLOAD_BYTES = int(os.getenv("AGENT_VOICE_MAX_STT_UPLOAD_BYTES") or str(25 * 1024 * 1024))
ALLOWED_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
SAFE_STT_GENERATION_OPTIONS = {"language", "verbose", "max_tokens", "chunk_duration", "initial_prompt"}
TTS_RESPONSE_FORMATS = {
    "wav": {"container": "wav", "media_type": "audio/wav"},
    "mp3": {"container": "mp3", "media_type": "audio/mpeg"},
    "opus": {"container": "ogg", "media_type": "audio/ogg"},
    "flac": {"container": "flac", "media_type": "audio/flac"},
}
NOTIFY_DEFAULT_TITLE = os.getenv("AGENT_VOICE_NOTIFY_DEFAULT_TITLE") or "PAI Notification"
NOTIFY_DEFAULT_MESSAGE = os.getenv("AGENT_VOICE_NOTIFY_DEFAULT_MESSAGE") or "Task completed"
NOTIFY_MAX_CHARS = int(os.getenv("AGENT_VOICE_NOTIFY_MAX_CHARS") or "1000")
NOTIFY_RATE_LIMIT = int(os.getenv("AGENT_VOICE_NOTIFY_RATE_LIMIT") or "10")
NOTIFY_RATE_WINDOW_SECONDS = float(os.getenv("AGENT_VOICE_NOTIFY_RATE_WINDOW_SECONDS") or "60")
NOTIFY_QUEUE_MAX_DEPTH = int(os.getenv("AGENT_VOICE_NOTIFY_QUEUE_MAX_DEPTH") or "3")
NOTIFY_PLAYBACK_TIMEOUT_SECONDS = float(os.getenv("AGENT_VOICE_NOTIFY_PLAYBACK_TIMEOUT_SECONDS") or "0")
NOTIFY_PLAYBACK_GRACE_SECONDS = float(os.getenv("AGENT_VOICE_NOTIFY_PLAYBACK_GRACE_SECONDS") or "2")
NOTIFY_OSASCRIPT_TIMEOUT_SECONDS = float(os.getenv("AGENT_VOICE_NOTIFY_OSASCRIPT_TIMEOUT_SECONDS") or "5")
NOTIFY_CORS_HEADERS = {
    "Access-Control-Allow-Origin": "http://localhost",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}

app = FastAPI(title="agent-voice", version="0.2.1")
_tts_model = None
_stt_model = None
_tts_model_lock = threading.Lock()
_tts_generation_lock = threading.Lock()
_stt_model_lock = threading.Lock()
_dropped_stt_options_lock = threading.Lock()
_notify_rate_lock = threading.Lock()
_notify_queue_lock = threading.Lock()
_notify_playback_lock = threading.Lock()
_logged_dropped_stt_options: set[tuple[str, ...]] = set()
_notify_request_counts: dict[str, tuple[int, float]] = {}
_notify_queue_depth = 0
_QWEN_TRANSFORMERS_CONFIG_WARNING = (
    "You are using a model of type `qwen3_tts` to instantiate a model of type ``."
)
_MLX_WHISPER_PROCESSOR_WARNING = r"Could not load WhisperProcessor: .*"


class AudioFormatError(RuntimeError):
    """Raised when requested response audio encoding cannot be produced."""


@dataclass(frozen=True)
class AudioActivity:
    duration_seconds: float
    active_seconds: float
    active_end_seconds: float


class _MessagePrefixFilter(logging.Filter):
    def __init__(self, prefix: str) -> None:
        super().__init__()
        self.prefix = prefix

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith(self.prefix)


@contextmanager
def _suppress_known_loader_noise():
    """Hide expected third-party loader warnings that agent-voice handles."""
    transformers_logger = logging.getLogger("transformers.configuration_utils")
    transformers_filter = _MessagePrefixFilter(_QWEN_TRANSFORMERS_CONFIG_WARNING)
    transformers_logger.addFilter(transformers_filter)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=_MLX_WHISPER_PROCESSOR_WARNING,
                category=UserWarning,
                module=r"mlx_audio\.stt\.models\.whisper\.whisper",
            )
            yield
    finally:
        transformers_logger.removeFilter(transformers_filter)


class RequestPayload(BaseModel):
    model: str = "qwen3-tts"
    input: str = Field(min_length=1)
    voice: str = "cyberpunk_cool"
    response_format: str = "wav"
    language: str = "English"
    instruct: str | None = Field(default=None, max_length=4000)
    max_tokens: int | None = Field(default=None, ge=1, le=TTS_REQUEST_MAX_TOKENS_LIMIT)


class MutePayload(BaseModel):
    muted: bool


class NotifyPayload(BaseModel):
    title: str | None = None
    message: str | None = None
    voice_enabled: bool = True
    voice_id: str | None = Field(default=None, max_length=200)
    voice_name: str | None = Field(default=None, max_length=200)
    instruct: str | None = Field(default=None, max_length=4000)
    language: str = "English"


class VoiceQueueFullError(RuntimeError):
    """Raised when queued local playback would pile up stale notifications."""


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
    if os.getenv("AGENT_VOICE_DISABLE_MODEL_LOAD") == "1":
        raise RuntimeError("model loading disabled")
    try:
        from mlx_audio.tts.utils import load_model as mlx_load_model
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError(
            "MLX audio runtime is not installed; run the installer or install the mlx extra"
        ) from exc

    started = time.perf_counter()
    print(f"Loading TTS model {TTS_MODEL_ID}...")
    with _suppress_known_loader_noise():
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
    if os.getenv("AGENT_VOICE_DISABLE_MODEL_LOAD") == "1":
        raise RuntimeError("model loading disabled")
    try:
        from mlx_audio.stt.utils import load_model as mlx_stt_load_model
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError(
            "MLX audio runtime is not installed; run the installer or install the mlx extra"
        ) from exc

    started = time.perf_counter()
    print(f"Loading STT model {STT_MODEL_ID}...")
    with _suppress_known_loader_noise():
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
        model._processor = _load_stt_processor(WhisperProcessor)
    except Exception as exc:  # pragma: no cover - depends on network/cache state
        raise RuntimeError(f"Failed to load Whisper processor {STT_PROCESSOR_ID}") from exc


def _load_stt_processor(processor_class: Any) -> Any:
    """Prefer cached processor metadata; fall back to Hub download for first use."""
    try:
        return processor_class.from_pretrained(STT_PROCESSOR_ID, local_files_only=True)
    except Exception:
        return processor_class.from_pretrained(STT_PROCESSOR_ID)


def generate_audio(
    text: str,
    instruct: str,
    language: str,
    response_format: str,
    max_tokens: int = TTS_MAX_TOKENS,
) -> bytes:
    """Generate audio bytes for a validated speech request."""
    model = get_tts_model()
    # The MLX/Qwen TTS runtime is process-global and has shown native crashes
    # under concurrent generation. Serialize synthesis while keeping health and
    # STT requests independent.
    with _tts_generation_lock:
        segments = _split_speech_text(text, TTS_MAX_SEGMENT_CHARS)
        chunks: list[np.ndarray] = []
        sample_rate = 24000
        output_sample_rate: int | None = None

        for segment in segments:
            for audio, sample_rate in _generate_audio_parts_for_segment(
                model=model,
                text=segment,
                instruct=instruct,
                language=language,
                max_tokens=max_tokens,
            ):
                if audio.size == 0:
                    continue
                if output_sample_rate is None:
                    output_sample_rate = sample_rate
                elif sample_rate != output_sample_rate:
                    raise RuntimeError(
                        f"TTS segments returned inconsistent sample rates: {output_sample_rate} and {sample_rate}"
                    )
                if chunks:
                    chunks.append(np.zeros(int(sample_rate * TTS_SEGMENT_SILENCE_SECONDS), dtype=audio.dtype))
                chunks.append(audio)

        if not chunks:
            return b""

        audio = np.concatenate(chunks)
        return _encode_audio(audio, output_sample_rate or sample_rate, response_format)


def _normalize_tts_response_format(response_format: str) -> str:
    normalized = response_format.strip().lower()
    if normalized not in TTS_RESPONSE_FORMATS:
        supported = ", ".join(sorted(TTS_RESPONSE_FORMATS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response_format; supported values: {supported}",
        )
    return normalized


def _encode_audio(audio: np.ndarray, sample_rate: int, response_format: str) -> bytes:
    normalized = _normalize_tts_response_format(response_format)
    audio = _prepare_audio_for_encoding(audio)
    if normalized in {"wav", "flac"}:
        output = io.BytesIO()
        sf.write(output, audio, sample_rate, format=normalized.upper())
        return output.getvalue()

    output = io.BytesIO()
    sf.write(output, audio, sample_rate, format="WAV")
    return _convert_wav_bytes(output.getvalue(), normalized)


def _prepare_audio_for_encoding(audio: np.ndarray) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.float32)
    if samples.size == 0:
        return samples

    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    if TTS_PEAK_LIMIT <= 0:
        return samples

    peak = float(np.max(np.abs(samples)))
    if peak > TTS_PEAK_LIMIT:
        samples = samples * (TTS_PEAK_LIMIT / peak)
    return samples


def _muted_audio(response_format: str) -> bytes:
    samples = np.zeros(int(24000 * 0.25), dtype=np.float32)
    return _encode_audio(samples, 24000, response_format)


def _convert_wav_bytes(wav_bytes: bytes, response_format: str) -> bytes:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise AudioFormatError(f"ffmpeg is required for response_format={response_format}")

    if response_format == "mp3":
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "wav", "-i", "pipe:0", "-f", "mp3", "pipe:1"]
    elif response_format == "opus":
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "wav",
            "-i",
            "pipe:0",
            "-acodec",
            "libopus",
            "-ac",
            "1",
            "-b:a",
            "64k",
            "-vbr",
            "off",
            "-f",
            "ogg",
            "pipe:1",
        ]
    else:  # pragma: no cover - protected by _normalize_tts_response_format
        raise RuntimeError(f"Unsupported response_format={response_format}")

    result = subprocess.run(cmd, input=wav_bytes, capture_output=True, timeout=FFMPEG_TIMEOUT_SECONDS)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise AudioFormatError(f"ffmpeg failed for response_format={response_format}: {stderr}")
    return result.stdout


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
    best_active_seconds = -1.0

    for attempt in range(attempts):
        audio, sample_rate, token_count = _generate_audio_segment_once(
            model=model,
            text=text,
            instruct=instruct,
            language=language,
            max_tokens=max_tokens,
            attempt=attempt,
        )
        audio = _trim_trailing_inactive_audio(audio, sample_rate)
        activity = _audio_activity(audio, sample_rate)
        if activity.active_seconds > best_active_seconds or (
            activity.active_seconds == best_active_seconds and audio.size > best_audio.size
        ):
            best_audio = audio
            best_sample_rate = sample_rate
            best_active_seconds = activity.active_seconds
        if not _is_suspiciously_short_audio(text, audio, sample_rate):
            return audio, sample_rate
        if attempt < attempts - 1:
            print(
                "Retrying suspiciously short TTS segment "
                f"(attempt={attempt + 1}, words={_word_count(text)}, "
                f"active_seconds={activity.active_seconds:.2f}, "
                f"duration_seconds={activity.duration_seconds:.2f}, "
                f"tokens={token_count})",
                file=sys.stderr,
            )

    return best_audio, best_sample_rate


def _generate_audio_parts_for_segment(
    model: Any,
    text: str,
    instruct: str,
    language: str,
    max_tokens: int,
) -> list[tuple[np.ndarray, int]]:
    audio, sample_rate = _generate_audio_segment(
        model=model,
        text=text,
        instruct=instruct,
        language=language,
        max_tokens=max_tokens,
    )
    if not _is_suspiciously_short_audio(text, audio, sample_rate):
        return [(audio, sample_rate)]

    fallback_segments = _split_speech_text_at_sentence_boundaries(text, TTS_MAX_SEGMENT_CHARS)
    if len(fallback_segments) <= 1:
        return [(audio, sample_rate)]

    print(
        "Falling back to sentence-level TTS after collapsed continuous segment "
        f"(words={_word_count(text)}, segments={len(fallback_segments)})",
        file=sys.stderr,
    )
    return [
        _generate_audio_segment(
            model=model,
            text=fallback_segment,
            instruct=instruct,
            language=language,
            max_tokens=max_tokens,
        )
        for fallback_segment in fallback_segments
    ]


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
    temperature, top_p, repetition_penalty = _tts_sampling_params(attempt)

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


def _tts_sampling_params(attempt: int) -> tuple[float, float, float]:
    if attempt <= 0:
        return TTS_TEMPERATURE, TTS_TOP_P, TTS_REPETITION_PENALTY
    return TTS_RETRY_TEMPERATURE, TTS_RETRY_TOP_P, TTS_RETRY_REPETITION_PENALTY


def _split_speech_text(text: str, max_segment_chars: int) -> list[str]:
    if max_segment_chars <= 0 or len(text) <= max_segment_chars:
        return [text]

    segments: list[str] = []
    current = ""
    for sentence in _split_speech_text_at_sentence_boundaries(text, max_segment_chars):
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


def _split_speech_text_at_sentence_boundaries(text: str, max_segment_chars: int) -> list[str]:
    segments: list[str] = []
    for sentence in re.split(r"(?<=[.!?;:])\s+", text):
        if not sentence:
            continue
        if max_segment_chars > 0 and len(sentence) > max_segment_chars:
            segments.extend(_split_long_speech_piece(sentence, max_segment_chars))
        else:
            segments.append(sentence)
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
    if words < TTS_SUSPICIOUS_MIN_WORDS or sample_rate <= 0 or audio.size == 0:
        return False
    activity = _audio_activity(audio, sample_rate)
    # The threshold is intentionally permissive; only retry obvious early-EOS or
    # speech-then-silence clips.
    min_reasonable_seconds = max(
        TTS_SUSPICIOUS_MIN_SECONDS,
        words / TTS_SUSPICIOUS_MAX_WORDS_PER_SECOND,
    )
    return activity.active_seconds < min_reasonable_seconds


def _trim_trailing_inactive_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate <= 0 or audio.size == 0:
        return audio
    activity = _audio_activity(audio, sample_rate)
    if activity.active_end_seconds <= 0:
        return audio

    keep_seconds = min(
        activity.duration_seconds,
        activity.active_end_seconds + TTS_SEGMENT_SILENCE_SECONDS,
    )
    if activity.duration_seconds - keep_seconds <= TTS_ACTIVITY_MAX_SILENT_GAP_SECONDS:
        return audio

    keep_samples = max(1, int(keep_seconds * sample_rate))
    return np.asarray(audio)[:keep_samples]


def _audio_activity(audio: np.ndarray, sample_rate: int) -> AudioActivity:
    if sample_rate <= 0 or audio.size == 0:
        return AudioActivity(duration_seconds=0.0, active_seconds=0.0, active_end_seconds=0.0)

    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)
    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)

    duration_seconds = samples.size / sample_rate
    window_samples = max(1, int(sample_rate * TTS_ACTIVITY_WINDOW_SECONDS))
    rms_values = []
    for start in range(0, samples.size, window_samples):
        window = samples[start : start + window_samples]
        if window.size:
            rms_values.append(float(np.sqrt(np.mean(window * window))))
    if not rms_values:
        return AudioActivity(
            duration_seconds=duration_seconds,
            active_seconds=0.0,
            active_end_seconds=0.0,
        )

    rms = np.asarray(rms_values, dtype=np.float32)
    threshold = max(TTS_ACTIVITY_MIN_RMS, float(np.max(rms)) * TTS_ACTIVITY_RELATIVE_RMS)
    active_indexes = np.flatnonzero(rms > threshold)
    if active_indexes.size == 0:
        return AudioActivity(
            duration_seconds=duration_seconds,
            active_seconds=0.0,
            active_end_seconds=0.0,
        )

    active_seconds = min(
        duration_seconds,
        _longest_active_span_windows(
            active_indexes,
            max_inactive_gap_windows=max(0, int(TTS_ACTIVITY_MAX_SILENT_GAP_SECONDS / TTS_ACTIVITY_WINDOW_SECONDS)),
        )
        * window_samples
        / sample_rate,
    )
    active_end_seconds = min(duration_seconds, (int(active_indexes[-1]) + 1) * window_samples / sample_rate)
    return AudioActivity(
        duration_seconds=duration_seconds,
        active_seconds=active_seconds,
        active_end_seconds=active_end_seconds,
    )


def _longest_active_span_windows(active_indexes: np.ndarray, max_inactive_gap_windows: int) -> int:
    """Return the longest active span, allowing natural short pauses."""
    longest = 1
    run_start = int(active_indexes[0])
    previous = run_start
    for raw_index in active_indexes[1:]:
        index = int(raw_index)
        if index - previous <= max_inactive_gap_windows + 1:
            longest = max(longest, index - run_start + 1)
        else:
            run_start = index
        previous = index
    return longest


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


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _notify_default_voice() -> str:
    configured = os.getenv("AGENT_VOICE_NOTIFY_DEFAULT_VOICE") or "cyberpunk_cool"
    return configured if configured in VOICE_DESIGNS else "cyberpunk_cool"


def _notify_pronunciations_path() -> Path:
    configured = os.getenv("AGENT_VOICE_PRONUNCIATIONS_PATH")
    if configured:
        return Path(configured).expanduser()
    return mute_state.state_dir() / "pronunciations.json"


def _load_pronunciation_rules() -> list[tuple[re.Pattern[str], str]]:
    path = _notify_pronunciations_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as exc:
        print(f"Failed to load pronunciation rules from {path}: {exc}", file=sys.stderr)
        return []

    replacements = data.get("replacements") if isinstance(data, dict) else None
    if not isinstance(replacements, list):
        return []

    rules: list[tuple[re.Pattern[str], str]] = []
    for item in replacements:
        if not isinstance(item, dict):
            continue
        term = item.get("term")
        phonetic = item.get("phonetic")
        if not isinstance(term, str) or not term or not isinstance(phonetic, str):
            continue
        rules.append((re.compile(rf"(?<!\w){re.escape(term)}(?!\w)"), phonetic))
    return rules


def _apply_pronunciations(text: str) -> str:
    result = text
    for pattern, phonetic in _load_pronunciation_rules():
        result = pattern.sub(phonetic, result)
    return result


def _sanitize_notify_text(value: str, field: str) -> str:
    if len(value) > NOTIFY_MAX_CHARS:
        raise HTTPException(status_code=400, detail=f"{field} too long (max {NOTIFY_MAX_CHARS} characters)")
    sanitized = (
        value.replace("\r", " ")
        .replace("\n", " ")
    )
    sanitized = re.sub(r"<script", "", sanitized, flags=re.IGNORECASE)
    sanitized = sanitized.replace("../", "")
    sanitized = re.sub(r"[;&|><`$\\]", "", sanitized)
    sanitized = re.sub(r"\*\*([^*]+)\*\*", r"\1", sanitized)
    sanitized = re.sub(r"\*([^*]+)\*", r"\1", sanitized)
    sanitized = re.sub(r"`([^`]+)`", r"\1", sanitized)
    sanitized = re.sub(r"#{1,6}\s+", "", sanitized)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    if not sanitized:
        raise HTTPException(status_code=400, detail=f"{field} contains no valid content after sanitization")
    return sanitized[:NOTIFY_MAX_CHARS]


def _escape_for_applescript(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _notify_client_id(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip() or "localhost"
    return request.client.host if request.client else "localhost"


def _notify_rate_limited(client_id: str) -> bool:
    if NOTIFY_RATE_LIMIT <= 0:
        return False

    now = time.time()
    with _notify_rate_lock:
        count, reset_time = _notify_request_counts.get(client_id, (0, 0.0))
        if now > reset_time:
            _notify_request_counts[client_id] = (1, now + NOTIFY_RATE_WINDOW_SECONDS)
            return False
        if count >= NOTIFY_RATE_LIMIT:
            return True
        _notify_request_counts[client_id] = (count + 1, reset_time)
        return False


def _notify_json(payload: dict[str, Any], status_code: int = 200) -> JSONResponse:
    return JSONResponse(content=payload, status_code=status_code, headers=NOTIFY_CORS_HEADERS)


def _display_desktop_notification(title: str, message: str) -> bool:
    if not _env_flag("AGENT_VOICE_NOTIFY_DESKTOP", True):
        return False

    osascript = shutil.which("osascript") or "/usr/bin/osascript"
    script = (
        f'display notification "{_escape_for_applescript(message)}" '
        f'with title "{_escape_for_applescript(title)}" sound name ""'
    )
    try:
        result = subprocess.run(
            [osascript, "-e", script],
            capture_output=True,
            timeout=NOTIFY_OSASCRIPT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"Notification display error: {exc}", file=sys.stderr)
        return False
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        print(f"Notification display error: {stderr}", file=sys.stderr)
        return False
    return True


def _run_notify_voice_queued(work: Any) -> None:
    global _notify_queue_depth

    with _notify_queue_lock:
        if _notify_queue_depth >= NOTIFY_QUEUE_MAX_DEPTH:
            raise VoiceQueueFullError("voice queue full; dropping stale voice notification")
        _notify_queue_depth += 1

    try:
        with _notify_playback_lock:
            work()
    finally:
        with _notify_queue_lock:
            _notify_queue_depth = max(0, _notify_queue_depth - 1)


def _wav_duration_seconds(wav_bytes: bytes) -> float | None:
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            frame_rate = wav.getframerate()
            if frame_rate <= 0:
                return None
            return wav.getnframes() / frame_rate
    except (EOFError, wave.Error):
        return None


def _notify_playback_timeout_seconds(wav_bytes: bytes) -> float | None:
    if NOTIFY_PLAYBACK_TIMEOUT_SECONDS > 0:
        return NOTIFY_PLAYBACK_TIMEOUT_SECONDS
    duration = _wav_duration_seconds(wav_bytes)
    if duration is None:
        return 60.0
    return max(1.0, duration + NOTIFY_PLAYBACK_GRACE_SECONDS)


def _play_audio_bytes(wav_bytes: bytes) -> None:
    afplay = shutil.which("afplay") or "/usr/bin/afplay"
    fd, path = tempfile.mkstemp(prefix="agent-voice-notify-", suffix=".wav")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(wav_bytes)
        result = subprocess.run(
            [afplay, path],
            capture_output=True,
            timeout=_notify_playback_timeout_seconds(wav_bytes),
        )
        if result.returncode != 0:
            raise RuntimeError(f"afplay exited with code {result.returncode}")
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"afplay timed out after {exc.timeout:g}s") from exc
    finally:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def _resolve_notify_voice(payload: NotifyPayload) -> tuple[str, bool]:
    requested = payload.voice_id or payload.voice_name
    custom_instruct = payload.instruct.strip() if payload.instruct else ""
    if requested and (requested in VOICE_DESIGNS or custom_instruct):
        return requested, False
    if requested:
        print(f"Unknown notify voice '{requested}'; falling back to {_notify_default_voice()}", file=sys.stderr)
        return _notify_default_voice(), True
    return _notify_default_voice(), False


def _generate_and_play_notification_audio(
    *,
    text: str,
    voice: str,
    instruct: str | None,
    language: str,
) -> None:
    speech_text = _apply_pronunciations(text)
    custom_instruct = instruct.strip() if instruct else ""
    voice_design = custom_instruct or VOICE_DESIGNS.get(voice)
    if not voice_design:
        raise RuntimeError(f"Unknown voice '{voice}'")

    audio = generate_audio(
        text=speech_text,
        instruct=voice_design,
        language=language,
        response_format="wav",
        max_tokens=TTS_MAX_TOKENS,
    )
    if not audio:
        raise RuntimeError("No audio generated")
    _play_audio_bytes(audio)


def _send_notification(payload: NotifyPayload) -> dict[str, Any]:
    title = _sanitize_notify_text(payload.title or NOTIFY_DEFAULT_TITLE, "title")
    message = _sanitize_notify_text(payload.message or NOTIFY_DEFAULT_MESSAGE, "message")
    voice, voice_fallback = _resolve_notify_voice(payload)

    desktop_displayed = _display_desktop_notification(title, message)
    muted = mute_state.is_muted()
    voice_played = False
    voice_error: str | None = None

    if payload.voice_enabled and not muted:
        try:
            _run_notify_voice_queued(
                lambda: _generate_and_play_notification_audio(
                    text=message,
                    voice=voice,
                    instruct=payload.instruct,
                    language=payload.language,
                )
            )
            voice_played = True
        except Exception as exc:
            print(f"Failed to generate/play speech: {exc}", file=sys.stderr)
            voice_error = str(exc) or "TTS generation failed"

    if payload.voice_enabled and voice_error:
        return {
            "status": "partial",
            "message": "Notification displayed, voice skipped",
            "voice_error": voice_error,
            "voice_played": False,
            "voice": voice,
            "voice_fallback": voice_fallback,
            "muted": muted,
            "desktop_displayed": desktop_displayed,
        }

    return {
        "status": "success",
        "message": "Notification sent",
        "voice_played": voice_played,
        "voice": voice,
        "voice_fallback": voice_fallback,
        "muted": muted,
        "desktop_displayed": desktop_displayed,
    }


def _notify_health() -> dict[str, Any]:
    host = os.getenv("AGENT_VOICE_HOST") or "127.0.0.1"
    port = int(os.getenv("AGENT_VOICE_PORT") or "8880")
    local_base_url = f"http://{host}:{port}"
    return {
        "status": "healthy",
        "port": port,
        "voice_system": "agent-voice",
        "default_voice": _notify_default_voice(),
        "upstream_url": f"{local_base_url}/v1/audio/speech",
        "upstream_health_url": f"{local_base_url}/v1/health",
        "pronunciation_rules": len(_load_pronunciation_rules()),
        "pronunciations_path": str(_notify_pronunciations_path()),
        "known_voices": list(VOICE_DESIGNS.keys()),
        "known_voices_source": "local",
        "voice_queue_depth": _notify_queue_depth,
        "voice_queue_max_depth": NOTIFY_QUEUE_MAX_DEPTH,
        "rate_limit": NOTIFY_RATE_LIMIT,
        "rate_window_seconds": NOTIFY_RATE_WINDOW_SECONDS,
        "desktop_notifications": _env_flag("AGENT_VOICE_NOTIFY_DESKTOP", True),
    }


@app.options("/notify")
@app.options("/notify/personality")
@app.options("/pai")
def notify_options() -> Response:
    return Response(status_code=204, headers=NOTIFY_CORS_HEADERS)


@app.get("/")
def root() -> Response:
    return Response(
        content="agent-voice - POST to /notify, /notify/personality, /pai, or /v1/audio/speech",
        media_type="text/plain",
    )


@app.get("/health")
def notify_health() -> JSONResponse:
    return _notify_json(_notify_health())


@app.post("/notify")
@app.post("/notify/personality")
@app.post("/pai")
def notify(payload: NotifyPayload, request: Request) -> JSONResponse:
    client_id = _notify_client_id(request)
    if _notify_rate_limited(client_id):
        return _notify_json({"status": "error", "message": "Rate limit exceeded"}, status_code=429)
    return _notify_json(_send_notification(payload))


@app.get("/v1/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "muted": mute_state.is_muted(),
        "model": "qwen3-tts",
        "tts_model_id": TTS_MODEL_ID,
        "tts_max_tokens": TTS_MAX_TOKENS,
        "stt_model_id": STT_MODEL_ID,
        "stt_processor_id": STT_PROCESSOR_ID,
        "voices": list(VOICE_DESIGNS.keys()),
    }


@app.get("/v1/mute")
def get_mute() -> dict[str, object]:
    return mute_state.read_state()


@app.post("/v1/mute")
def set_mute(request: MutePayload) -> dict[str, object]:
    try:
        return mute_state.set_muted(request.muted)
    except mute_state.MuteStateLockedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/v1/mute/toggle")
def toggle_mute() -> dict[str, object]:
    try:
        return mute_state.toggle_muted()
    except mute_state.MuteStateLockedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/v1/audio/speech")
def audio_speech(request: RequestPayload) -> Response:
    if request.model not in {"qwen3-tts", TTS_MODEL_ID}:
        raise HTTPException(status_code=400, detail="Unsupported model")

    response_format = _normalize_tts_response_format(request.response_format)

    text = request.input.strip()
    if not text:
        raise HTTPException(status_code=400, detail="input must not be empty")

    if mute_state.is_muted():
        try:
            audio = _muted_audio(response_format)
        except AudioFormatError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return Response(
            content=audio,
            media_type=TTS_RESPONSE_FORMATS[response_format]["media_type"],
            headers={"X-Agent-Voice-Muted": "true", "X-Generation-Time": "0.000"},
        )

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
            response_format=response_format,
            max_tokens=request.max_tokens or TTS_MAX_TOKENS,
        )
    except AudioFormatError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Speech generation failed") from exc

    if not audio:
        raise HTTPException(status_code=500, detail="No audio generated")

    elapsed = time.perf_counter() - started
    return Response(
        content=audio,
        media_type=TTS_RESPONSE_FORMATS[response_format]["media_type"],
        headers={"X-Generation-Time": f"{elapsed:.3f}"},
    )


@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    file: UploadFile = File(...),
    model: str = Form(STT_MODEL_ID),
    language: str | None = Form(None),
    verbose: bool | None = Form(None),
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
                "verbose": True if verbose else None,
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
    port = int(os.getenv("AGENT_VOICE_PORT") or "8880")
    uvicorn.run("agent_voice.server:app", host=host, port=port)


def _server_bind_config() -> tuple[str, bool]:
    return os.getenv("AGENT_VOICE_HOST") or "127.0.0.1", os.getenv("AGENT_VOICE_ALLOW_REMOTE") == "1"


if __name__ == "__main__":
    main()

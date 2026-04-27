"""FastAPI server for the codex-tts OpenAI-compatible speech subset."""

from __future__ import annotations

import io
import os
import time
import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .voices import VOICE_DESIGNS

TTS_MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"

app = FastAPI(title="codex-tts", version="0.1.0")
_tts_model = None


class RequestPayload(BaseModel):
    model: str = "qwen3-tts"
    input: str = Field(min_length=1)
    voice: str = "peng_mythic"
    response_format: str = "wav"
    language: str = "English"
    instruct: str | None = None


def get_tts_model():
    """Load the MLX model lazily; health checks must stay cheap."""
    global _tts_model
    if _tts_model is None:
        if os.getenv("CODEX_TTS_DISABLE_MODEL_LOAD") == "1":
            raise RuntimeError("model loading disabled")
        try:
            from mlx_audio.tts.utils import load_model as mlx_load_model
        except Exception as exc:  # pragma: no cover - depends on optional runtime
            raise RuntimeError(
                "MLX audio runtime is not installed; run the installer or install the mlx extra"
            ) from exc

        started = time.perf_counter()
        print(f"Loading TTS model {TTS_MODEL_ID}...")
        _tts_model = mlx_load_model(TTS_MODEL_ID)
        print(f"TTS model loaded in {time.perf_counter() - started:.1f}s")
    return _tts_model


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


@app.get("/v1/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model": "qwen3-tts",
        "tts_model_id": TTS_MODEL_ID,
        "voices": list(VOICE_DESIGNS.keys()),
    }


@app.post("/v1/audio/speech")
def audio_speech(request: RequestPayload) -> Response:
    if request.response_format != "wav":
        raise HTTPException(status_code=400, detail="Unsupported response_format; only wav is supported")

    voice_design = request.instruct or VOICE_DESIGNS.get(request.voice)
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
            text=request.input.strip(),
            instruct=voice_design,
            language=request.language,
            response_format=request.response_format,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not audio:
        raise HTTPException(status_code=500, detail="No audio generated")

    elapsed = time.perf_counter() - started
    return Response(
        content=audio,
        media_type="audio/wav",
        headers={"X-Generation-Time": f"{elapsed:.3f}"},
    )


def main() -> None:
    host = os.getenv("CODEX_TTS_HOST", "127.0.0.1")
    port = int(os.getenv("CODEX_TTS_PORT", "8880"))
    uvicorn.run("codex_tts.server:app", host=host, port=port)


if __name__ == "__main__":
    main()

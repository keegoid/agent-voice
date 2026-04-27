from __future__ import annotations

import json
from typing import Any

import pytest

from tests.helpers.public_contract import PUBLIC_VOICES, locate_fastapi_app, patch_generation

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient


def _json_response_field(data: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in data:
            return data[name]
    raise AssertionError(f"missing any of fields: {names}; got {data}")


def test_health_reports_status_model_and_public_voices_without_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    app = locate_fastapi_app()
    import codex_tts.server as server

    def fail_if_loaded(*_args: Any, **_kwargs: Any) -> bytes:
        raise AssertionError("health must not trigger speech generation or model loading")

    patch_generation(monkeypatch, app, fail_if_loaded)
    monkeypatch.setattr(server, "get_stt_model", fail_if_loaded)
    client = TestClient(app)

    response = client.get("/v1/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert _json_response_field(data, "model", "model_id", "tts_model_id")
    assert data["stt_model_id"]
    voices = set(_json_response_field(data, "voices", "available_voices", "public_voices"))
    assert PUBLIC_VOICES <= voices


def test_speech_rejects_unknown_voice_without_instruct() -> None:
    client = TestClient(locate_fastapi_app())

    response = client.post(
        "/v1/audio/speech",
        json={"input": "hello", "voice": "not_a_public_voice", "response_format": "wav"},
    )

    assert response.status_code == 400


def test_speech_rejects_whitespace_only_input() -> None:
    client = TestClient(locate_fastapi_app())

    response = client.post(
        "/v1/audio/speech",
        json={"input": "   ", "voice": "cyberpunk_cool", "response_format": "wav"},
    )

    assert response.status_code == 400


def test_speech_accepts_custom_voice_when_instruct_is_provided(monkeypatch: pytest.MonkeyPatch) -> None:
    app = locate_fastapi_app()

    def fake_generation(*_args: Any, **_kwargs: Any) -> bytes:
        return b"RIFFfakeWAVEfmt "

    patch_generation(monkeypatch, app, fake_generation)
    client = TestClient(app)

    response = client.post(
        "/v1/audio/speech",
        json={
            "input": "hello",
            "voice": "custom_contract_voice",
            "instruct": "Speak warmly and clearly.",
            "response_format": "wav",
        },
    )

    assert response.status_code != 400


def test_speech_rejects_unsupported_response_format() -> None:
    client = TestClient(locate_fastapi_app())

    response = client.post(
        "/v1/audio/speech",
        json={"input": "hello", "voice": "cyberpunk_cool", "response_format": "mp3"},
    )

    assert response.status_code == 400


def test_speech_returns_500_when_generation_produces_no_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    app = locate_fastapi_app()

    def no_audio(*_args: Any, **_kwargs: Any) -> bytes:
        return b""

    patch_generation(monkeypatch, app, no_audio)
    client = TestClient(app)

    response = client.post(
        "/v1/audio/speech",
        json={"input": "hello", "voice": "cyberpunk_cool", "response_format": "wav"},
    )

    assert response.status_code == 500


def test_transcription_rejects_unsupported_model_without_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    import codex_tts.server as server

    def fail_if_loaded(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("unsupported model must be rejected before model loading")

    monkeypatch.setattr(server, "get_stt_model", fail_if_loaded)
    client = TestClient(locate_fastapi_app())

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "not-a-supported-stt-model"},
        files={"file": ("sample.wav", b"RIFFfakeWAVEfmt ", "audio/wav")},
    )

    assert response.status_code == 400


def test_transcription_returns_buffered_ndjson_from_mock_model(monkeypatch: pytest.MonkeyPatch) -> None:
    import codex_tts.server as server

    class FakeChunk:
        text = " final"
        start_time = 0.0
        end_time = 1.0
        is_final = True
        language = "en"

    class FakeSttModel:
        def generate(self, path: str, language: str | None = None, **_kwargs: Any) -> list[Any]:
            assert path.endswith(".wav")
            assert language == "en"
            return [{"text": "hello fig", "language": language}, FakeChunk()]

    monkeypatch.setattr(server, "get_stt_model", lambda: FakeSttModel())
    client = TestClient(locate_fastapi_app())

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "mlx-community/whisper-large-v3-mlx", "language": "en"},
        files={"file": ("sample.wav", b"RIFFfakeWAVEfmt ", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    lines = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert len(lines) == 2
    assert lines[0]["text"] == "hello fig"
    assert lines[0]["language"] == "en"
    assert lines[1]["text"] == " final"
    assert lines[1]["is_final"] is True


def test_transcription_rejects_short_model_alias_without_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    import codex_tts.server as server

    def fail_if_loaded(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("model aliases must be rejected before model loading")

    monkeypatch.setattr(server, "get_stt_model", fail_if_loaded)
    client = TestClient(locate_fastapi_app())

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-large-v3-mlx"},
        files={"file": ("sample.wav", b"RIFFfakeWAVEfmt ", "audio/wav")},
    )

    assert response.status_code == 400


def test_transcription_rejects_upload_one_byte_over_limit_without_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    import codex_tts.server as server

    def fail_if_loaded(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("oversized upload must be rejected before model loading")

    monkeypatch.setattr(server, "MAX_STT_UPLOAD_BYTES", 8)
    monkeypatch.setattr(server, "get_stt_model", fail_if_loaded)
    client = TestClient(locate_fastapi_app())

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "mlx-community/whisper-large-v3-mlx"},
        files={"file": ("sample.wav", b"012345678", "audio/wav")},
    )

    assert response.status_code == 413


def test_transcription_accepts_upload_at_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    import codex_tts.server as server

    class FakeSttModel:
        def generate(self, path: str, **_kwargs: Any) -> dict[str, str]:
            return {"text": "limit ok"}

    monkeypatch.setattr(server, "MAX_STT_UPLOAD_BYTES", 8)
    monkeypatch.setattr(server, "get_stt_model", lambda: FakeSttModel())
    client = TestClient(locate_fastapi_app())

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "mlx-community/whisper-large-v3-mlx"},
        files={"file": ("sample.wav", b"01234567", "audio/wav")},
    )

    assert response.status_code == 200
    lines = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert lines == [{"text": "limit ok"}]

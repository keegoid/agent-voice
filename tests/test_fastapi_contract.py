from __future__ import annotations

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

    def fail_if_loaded(*_args: Any, **_kwargs: Any) -> bytes:
        raise AssertionError("health must not trigger speech generation or model loading")

    patch_generation(monkeypatch, app, fail_if_loaded)
    client = TestClient(app)

    response = client.get("/v1/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert _json_response_field(data, "model", "model_id", "tts_model_id")
    voices = set(_json_response_field(data, "voices", "available_voices", "public_voices"))
    assert PUBLIC_VOICES <= voices


def test_speech_rejects_unknown_voice_without_instruct() -> None:
    client = TestClient(locate_fastapi_app())

    response = client.post(
        "/v1/audio/speech",
        json={"input": "hello", "voice": "not_a_public_voice", "response_format": "wav"},
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
        json={"input": "hello", "voice": "peng_mythic", "response_format": "mp3"},
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
        json={"input": "hello", "voice": "peng_mythic", "response_format": "wav"},
    )

    assert response.status_code == 500

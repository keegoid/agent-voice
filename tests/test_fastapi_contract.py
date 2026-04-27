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
    import agent_voice.server as server

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
    assert data["stt_processor_id"]
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


def test_speech_accepts_long_input_without_character_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    app = locate_fastapi_app()

    def fake_generation(*_args: Any, **_kwargs: Any) -> bytes:
        return b"RIFFfakeWAVEfmt "

    patch_generation(monkeypatch, app, fake_generation)
    client = TestClient(app)

    response = client.post(
        "/v1/audio/speech",
        json={"input": "long agent summary. " * 500, "voice": "cyberpunk_cool", "response_format": "wav"},
    )

    assert response.status_code == 200


def test_generate_audio_uses_configured_tts_token_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_voice.server as server

    seen: dict[str, Any] = {}

    class FakeResult:
        audio = [0.0] * 2400
        sample_rate = 24000
        token_count = 100

    class FakeModel:
        def generate_voice_design(self, **kwargs: Any) -> list[FakeResult]:
            seen.update(kwargs)
            return [FakeResult()]

    monkeypatch.setattr(server, "get_tts_model", lambda: FakeModel())

    audio = server.generate_audio(
        text="short token budget test",
        instruct="clear voice",
        language="English",
        response_format="wav",
        max_tokens=32123,
    )

    assert audio.startswith(b"RIFF")
    assert seen["max_tokens"] == 32123


def test_split_speech_text_bounds_single_long_word() -> None:
    import agent_voice.server as server

    segments = server._split_speech_text("x" * 2505, 1000)

    assert [len(segment) for segment in segments] == [1000, 1000, 505]


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
    import agent_voice.server as server

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


def test_stt_loader_attaches_fallback_whisper_processor(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    import agent_voice.server as server

    class FakeModel:
        _processor = None

    class FakeProcessor:
        @staticmethod
        def from_pretrained(model_id: str) -> str:
            assert model_id == server.STT_PROCESSOR_ID
            return "processor"

    fake_mlx_audio = types.ModuleType("mlx_audio")
    fake_stt = types.ModuleType("mlx_audio.stt")
    fake_utils = types.ModuleType("mlx_audio.stt.utils")
    fake_utils.load_model = lambda _model_id: FakeModel()
    fake_stt.utils = fake_utils
    fake_mlx_audio.stt = fake_stt
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.WhisperProcessor = FakeProcessor
    monkeypatch.setitem(sys.modules, "mlx_audio", fake_mlx_audio)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt", fake_stt)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.delenv("AGENT_VOICE_DISABLE_MODEL_LOAD", raising=False)
    monkeypatch.delenv("CODEX_TTS_DISABLE_MODEL_LOAD", raising=False)

    model = server._load_stt_model()

    assert model._processor == "processor"


def test_stt_filter_drops_unsafe_decode_options_with_var_kwargs() -> None:
    import agent_voice.server as server

    class FakeWhisperLikeModel:
        def generate(self, audio: str, *, language: str | None = None, **decode_options: Any) -> str:
            return audio

    filtered = server._filter_generation_kwargs(
        FakeWhisperLikeModel(),
        {
            "language": "en",
            "chunk_duration": 30.0,
            "frame_threshold": 25,
            "prefill_step_size": 2048,
            "initial_prompt": "local context",
        },
    )

    assert filtered == {
        "language": "en",
        "chunk_duration": 30.0,
        "initial_prompt": "local context",
    }


def test_agent_voice_allow_remote_must_match_agent_voice_host(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_voice.server as server

    monkeypatch.setenv("AGENT_VOICE_HOST", "0.0.0.0")
    monkeypatch.delenv("AGENT_VOICE_ALLOW_REMOTE", raising=False)
    monkeypatch.setenv("CODEX_TTS_ALLOW_REMOTE", "1")

    host, allow_remote = server._server_bind_config()

    assert host == "0.0.0.0"
    assert allow_remote is False


def test_transcription_returns_buffered_ndjson_from_mock_model(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_voice.server as server

    class FakeChunk:
        text = " final"
        start_time = 0.0
        end_time = 1.0
        is_final = True
        language = "en"

    class FakeSttModel:
        def generate(
            self,
            path: str,
            language: str | None = None,
            initial_prompt: str | None = None,
            **kwargs: Any,
        ) -> list[Any]:
            assert path.endswith(".wav")
            assert language == "en"
            assert initial_prompt == "legacy context"
            assert "frame_threshold" not in kwargs
            assert "prefill_step_size" not in kwargs
            return [{"text": "hello fig", "language": language}, FakeChunk()]

    monkeypatch.setattr(server, "get_stt_model", lambda: FakeSttModel())
    client = TestClient(locate_fastapi_app())

    response = client.post(
        "/v1/audio/transcriptions",
        data={
            "model": "mlx-community/whisper-large-v3-mlx",
            "language": "en",
            "context": "legacy context",
            "frame_threshold": "25",
            "prefill_step_size": "2048",
        },
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
    import agent_voice.server as server

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
    import agent_voice.server as server

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
    import agent_voice.server as server

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

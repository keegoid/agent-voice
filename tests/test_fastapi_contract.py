from __future__ import annotations

import concurrent.futures
import io
import json
import logging
import os
import threading
import time
import warnings
import wave
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
    assert data["muted"] is False
    assert _json_response_field(data, "model", "model_id", "tts_model_id")
    assert data["stt_model_id"]
    assert data["stt_processor_id"]
    voices = set(_json_response_field(data, "voices", "available_voices", "public_voices"))
    assert PUBLIC_VOICES <= voices


def test_mute_endpoints_persist_state(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_VOICE_MUTE_STATE", str(tmp_path / "mute.json"))
    monkeypatch.delenv("AGENT_VOICE_MUTED", raising=False)
    client = TestClient(locate_fastapi_app())

    assert client.get("/v1/mute").json()["muted"] is False

    enabled = client.post("/v1/mute", json={"muted": True})
    assert enabled.status_code == 200
    assert enabled.json()["muted"] is True
    assert json.loads((tmp_path / "mute.json").read_text(encoding="utf-8"))["muted"] is True

    toggled = client.post("/v1/mute/toggle")
    assert toggled.status_code == 200
    assert toggled.json()["muted"] is False


def test_mute_endpoints_reject_writes_when_env_override_is_set(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = tmp_path / "mute.json"
    monkeypatch.setenv("AGENT_VOICE_MUTE_STATE", str(state))
    monkeypatch.setenv("AGENT_VOICE_MUTED", "true")
    client = TestClient(locate_fastapi_app())

    assert client.get("/v1/mute").json()["source"] == "env"

    direct = client.post("/v1/mute", json={"muted": False})
    toggled = client.post("/v1/mute/toggle")

    assert direct.status_code == 409
    assert toggled.status_code == 409
    assert not state.exists()


def test_speech_returns_silence_without_generation_when_muted(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_voice.server as server

    monkeypatch.setenv("AGENT_VOICE_MUTE_STATE", str(tmp_path / "mute.json"))
    monkeypatch.delenv("AGENT_VOICE_MUTED", raising=False)
    server.mute_state.set_muted(True, source="test")

    def fail_if_generated(*_args: Any, **_kwargs: Any) -> bytes:
        raise AssertionError("muted speech must not generate model audio")

    monkeypatch.setattr(server, "generate_audio", fail_if_generated)
    client = TestClient(locate_fastapi_app())

    response = client.post(
        "/v1/audio/speech",
        json={"input": "hello", "voice": "not_a_public_voice", "response_format": "wav"},
    )

    assert response.status_code == 200
    assert response.headers["x-agent-voice-muted"] == "true"
    with wave.open(io.BytesIO(response.content), "rb") as wav:
        assert wav.getnframes() > 0
        assert wav.getframerate() == 24000


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


def test_speech_accepts_hermes_mp3_response_format(monkeypatch: pytest.MonkeyPatch) -> None:
    app = locate_fastapi_app()
    seen: dict[str, Any] = {}

    def fake_generation(*_args: Any, **kwargs: Any) -> bytes:
        seen.update(kwargs)
        return b"ID3fake-mp3"

    patch_generation(monkeypatch, app, fake_generation)
    client = TestClient(app)

    response = client.post(
        "/v1/audio/speech",
        json={"input": "hello", "voice": "cyberpunk_cool", "response_format": "mp3"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/mpeg")
    assert response.content == b"ID3fake-mp3"
    assert seen["response_format"] == "mp3"


def test_speech_accepts_hermes_opus_response_format(monkeypatch: pytest.MonkeyPatch) -> None:
    app = locate_fastapi_app()

    def fake_generation(*_args: Any, **_kwargs: Any) -> bytes:
        return b"OggSfake-opus"

    patch_generation(monkeypatch, app, fake_generation)
    client = TestClient(app)

    response = client.post(
        "/v1/audio/speech",
        json={"input": "hello", "voice": "cyberpunk_cool", "response_format": "opus"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/ogg")
    assert response.content == b"OggSfake-opus"


def test_generate_audio_transcodes_mp3_with_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
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

    def fake_convert(wav_bytes: bytes, response_format: str) -> bytes:
        assert wav_bytes.startswith(b"RIFF")
        assert response_format == "mp3"
        return b"ID3transcoded"

    monkeypatch.setattr(server, "get_tts_model", lambda: FakeModel())
    monkeypatch.setattr(server, "_convert_wav_bytes", fake_convert)

    audio = server.generate_audio(
        text="short mp3 transcode test",
        instruct="clear voice",
        language="English",
        response_format="mp3",
        max_tokens=123,
    )

    assert audio == b"ID3transcoded"
    assert seen["max_tokens"] == 123


def test_speech_returns_503_when_ffmpeg_is_missing_for_mp3(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_voice.server as server

    class FakeResult:
        audio = [0.0] * 2400
        sample_rate = 24000
        token_count = 100

    class FakeModel:
        def generate_voice_design(self, **_kwargs: Any) -> list[FakeResult]:
            return [FakeResult()]

    monkeypatch.setattr(server, "get_tts_model", lambda: FakeModel())
    monkeypatch.setattr(server.shutil, "which", lambda _name: None)
    client = TestClient(locate_fastapi_app())

    response = client.post(
        "/v1/audio/speech",
        json={"input": "hello", "voice": "cyberpunk_cool", "response_format": "mp3"},
    )

    assert response.status_code == 503
    assert "ffmpeg is required" in response.text


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


def test_generate_audio_serializes_shared_tts_model(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_voice.server as server

    state = {"active": False, "overlap": False, "calls": 0}
    guard = threading.Lock()
    start = threading.Barrier(2)

    class FakeResult:
        audio = [0.0] * 2400
        sample_rate = 24000
        token_count = 100

    class FakeModel:
        def generate_voice_design(self, **_kwargs: Any) -> list[FakeResult]:
            with guard:
                if state["active"]:
                    state["overlap"] = True
                state["active"] = True
                state["calls"] += 1
            try:
                time.sleep(0.05)
                return [FakeResult()]
            finally:
                with guard:
                    state["active"] = False

    monkeypatch.setattr(server, "get_tts_model", lambda: FakeModel())

    def worker(text: str) -> bytes:
        start.wait(timeout=2)
        return server.generate_audio(
            text=text,
            instruct="clear voice",
            language="English",
            response_format="wav",
            max_tokens=123,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(worker, "first serialized request"),
            pool.submit(worker, "second serialized request"),
        ]
        results = [future.result(timeout=5) for future in futures]

    assert all(result.startswith(b"RIFF") for result in results)
    assert state["calls"] == 2
    assert state["overlap"] is False


def test_generate_audio_retries_suspiciously_short_status_clip(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_voice.server as server

    calls = 0

    class FakeResult:
        sample_rate = 24000
        token_count = 100

        def __init__(self, samples: int) -> None:
            self.audio = [0.0] * samples

    class FakeModel:
        def generate_voice_design(self, **_kwargs: Any) -> list[FakeResult]:
            nonlocal calls
            calls += 1
            if calls == 1:
                return [FakeResult(1200)]
            return [FakeResult(48000)]

    monkeypatch.setattr(server, "get_tts_model", lambda: FakeModel())

    audio = server.generate_audio(
        text="Short status should finish cleanly",
        instruct="clear voice",
        language="English",
        response_format="wav",
        max_tokens=123,
    )

    assert audio.startswith(b"RIFF")
    assert calls == 2


def test_generate_audio_uses_stable_sampling_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_voice.server as server

    seen: list[dict[str, Any]] = []

    class FakeResult:
        audio = server.np.full(24000 * 3, 0.1, dtype=server.np.float32)
        sample_rate = 24000
        token_count = 100

    class FakeModel:
        def generate_voice_design(self, **kwargs: Any) -> list[FakeResult]:
            seen.append(kwargs)
            return [FakeResult()]

    monkeypatch.setattr(server, "get_tts_model", lambda: FakeModel())

    audio = server.generate_audio(
        text="Stable sampling should avoid noisy voice outliers",
        instruct="clear voice",
        language="English",
        response_format="wav",
        max_tokens=123,
    )

    assert audio.startswith(b"RIFF")
    assert seen[0]["temperature"] == server.TTS_TEMPERATURE
    assert seen[0]["top_p"] == server.TTS_TOP_P
    assert seen[0]["repetition_penalty"] == server.TTS_REPETITION_PENALTY


def test_generate_audio_uses_more_conservative_retry_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_voice.server as server

    seen: list[dict[str, Any]] = []

    class FakeResult:
        sample_rate = 24000
        token_count = 100

        def __init__(self, audio: Any) -> None:
            self.audio = audio

    class FakeModel:
        def generate_voice_design(self, **kwargs: Any) -> list[FakeResult]:
            seen.append(kwargs)
            if len(seen) == 1:
                return [FakeResult(server.np.zeros(2400, dtype=server.np.float32))]
            return [FakeResult(server.np.full(24000 * 3, 0.1, dtype=server.np.float32))]

    monkeypatch.setattr(server, "get_tts_model", lambda: FakeModel())

    audio = server.generate_audio(
        text="Retry sampling should reduce noisy collapsed output",
        instruct="clear voice",
        language="English",
        response_format="wav",
        max_tokens=123,
    )

    assert audio.startswith(b"RIFF")
    assert len(seen) == 2
    assert seen[1]["temperature"] == server.TTS_RETRY_TEMPERATURE
    assert seen[1]["top_p"] == server.TTS_RETRY_TOP_P
    assert seen[1]["repetition_penalty"] == server.TTS_RETRY_REPETITION_PENALTY


def test_encode_audio_sanitizes_and_limits_peak() -> None:
    import agent_voice.server as server

    samples = server.np.array(
        [0.0, 2.0, -2.0, server.np.nan, server.np.inf, -server.np.inf],
        dtype=server.np.float32,
    )

    encoded = server._encode_audio(samples, 24000, "wav")
    decoded, sample_rate = server.sf.read(io.BytesIO(encoded), dtype="float32")

    assert sample_rate == 24000
    assert server.np.isfinite(decoded).all()
    assert float(server.np.max(server.np.abs(decoded))) <= server.TTS_PEAK_LIMIT + 1 / 32768


def test_generate_audio_retries_speech_followed_by_silence(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_voice.server as server

    calls = 0
    sample_rate = 24000

    class FakeResult:
        sample_rate = 24000
        token_count = 100

        def __init__(self, audio: Any) -> None:
            self.audio = audio

    class FakeModel:
        def generate_voice_design(self, **_kwargs: Any) -> list[FakeResult]:
            nonlocal calls
            calls += 1
            if calls == 1:
                clipped = server.np.concatenate(
                    [
                        server.np.full(sample_rate, 0.1, dtype=server.np.float32),
                        server.np.zeros(sample_rate * 10, dtype=server.np.float32),
                    ]
                )
                return [FakeResult(clipped)]
            complete = server.np.full(sample_rate * 5, 0.1, dtype=server.np.float32)
            return [FakeResult(complete)]

    monkeypatch.setattr(server, "get_tts_model", lambda: FakeModel())

    audio = server.generate_audio(
        text="This status sentence should not stop speaking after the first clause.",
        instruct="clear voice",
        language="English",
        response_format="wav",
        max_tokens=123,
    )

    with wave.open(io.BytesIO(audio), "rb") as wav:
        duration = wav.getnframes() / wav.getframerate()

    assert calls == 2
    assert duration == pytest.approx(5.0)


def test_generate_audio_retries_interrupted_speech_with_late_blip(monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_voice.server as server

    calls = 0
    sample_rate = 24000

    class FakeResult:
        sample_rate = 24000
        token_count = 100

        def __init__(self, audio: Any) -> None:
            self.audio = audio

    class FakeModel:
        def generate_voice_design(self, **_kwargs: Any) -> list[FakeResult]:
            nonlocal calls
            calls += 1
            if calls == 1:
                interrupted = server.np.concatenate(
                    [
                        server.np.full(sample_rate // 2, 0.1, dtype=server.np.float32),
                        server.np.zeros(sample_rate * 8, dtype=server.np.float32),
                        server.np.full(sample_rate // 2, 0.1, dtype=server.np.float32),
                    ]
                )
                return [FakeResult(interrupted)]
            complete = server.np.full(sample_rate * 5, 0.1, dtype=server.np.float32)
            return [FakeResult(complete)]

    monkeypatch.setattr(server, "get_tts_model", lambda: FakeModel())

    audio = server.generate_audio(
        text="This status sentence should not be counted as complete by a late audio blip.",
        instruct="clear voice",
        language="English",
        response_format="wav",
        max_tokens=123,
    )

    with wave.open(io.BytesIO(audio), "rb") as wav:
        duration = wav.getnframes() / wav.getframerate()

    assert calls == 2
    assert duration == pytest.approx(5.0)


def test_split_speech_text_bounds_single_long_word() -> None:
    import agent_voice.server as server

    segments = server._split_speech_text("x" * 2505, 1000)

    assert [len(segment) for segment in segments] == [1000, 1000, 505]


def test_split_speech_text_keeps_short_multi_sentence_text_together() -> None:
    import agent_voice.server as server

    segments = server._split_speech_text(
        "First sentence should synthesize alone. Second sentence should not inherit its noise tail.",
        1200,
    )

    assert segments == [
        "First sentence should synthesize alone. Second sentence should not inherit its noise tail.",
    ]


def test_generate_audio_uses_continuous_generation_for_short_multi_sentence_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_voice.server as server

    seen_texts: list[str] = []
    sample_rate = 24000

    class FakeResult:
        sample_rate = 24000
        token_count = 100

        def __init__(self, audio: Any) -> None:
            self.audio = audio

    class FakeModel:
        def generate_voice_design(self, **kwargs: Any) -> list[FakeResult]:
            seen_texts.append(kwargs["text"])
            return [FakeResult(server.np.full(sample_rate * 5, 0.1, dtype=server.np.float32))]

    text = "First sentence should stay in one voice. Second sentence should share that voice."
    monkeypatch.setattr(server, "get_tts_model", lambda: FakeModel())

    audio = server.generate_audio(
        text=text,
        instruct="clear voice",
        language="English",
        response_format="wav",
        max_tokens=123,
    )

    assert audio.startswith(b"RIFF")
    assert seen_texts == [text]


def test_generate_audio_falls_back_to_sentences_only_after_continuous_collapse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_voice.server as server

    seen_texts: list[str] = []
    sample_rate = 24000
    text = "First sentence should synthesize cleanly. Second sentence should also synthesize cleanly."

    class FakeResult:
        sample_rate = 24000
        token_count = 100

        def __init__(self, audio: Any) -> None:
            self.audio = audio

    class FakeModel:
        def generate_voice_design(self, **kwargs: Any) -> list[FakeResult]:
            seen_texts.append(kwargs["text"])
            if kwargs["text"] == text:
                collapsed = server.np.concatenate(
                    [
                        server.np.full(sample_rate // 2, 0.1, dtype=server.np.float32),
                        server.np.zeros(sample_rate * 10, dtype=server.np.float32),
                    ]
                )
                return [FakeResult(collapsed)]
            return [FakeResult(server.np.full(sample_rate * 2, 0.1, dtype=server.np.float32))]

    monkeypatch.setattr(server, "get_tts_model", lambda: FakeModel())
    monkeypatch.setattr(server, "TTS_GENERATION_ATTEMPTS", 2)

    audio = server.generate_audio(
        text=text,
        instruct="clear voice",
        language="English",
        response_format="wav",
        max_tokens=123,
    )

    with wave.open(io.BytesIO(audio), "rb") as wav:
        duration = wav.getnframes() / wav.getframerate()

    assert seen_texts == [
        text,
        text,
        "First sentence should synthesize cleanly.",
        "Second sentence should also synthesize cleanly.",
    ]
    assert duration == pytest.approx(4.18)


def test_speech_rejects_unsupported_response_format() -> None:
    client = TestClient(locate_fastapi_app())

    response = client.post(
        "/v1/audio/speech",
        json={"input": "hello", "voice": "cyberpunk_cool", "response_format": "ogg"},
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
        def from_pretrained(model_id: str, **kwargs: Any) -> str:
            assert model_id == server.STT_PROCESSOR_ID
            assert kwargs == {"local_files_only": True}
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

    model = server._load_stt_model()

    assert model._processor == "processor"


def test_known_loader_noise_is_suppressed(caplog: pytest.LogCaptureFixture) -> None:
    import agent_voice.server as server

    transformers_logger = logging.getLogger("transformers.configuration_utils")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with caplog.at_level(logging.WARNING, logger="transformers.configuration_utils"):
            with server._suppress_known_loader_noise():
                transformers_logger.warning(server._QWEN_TRANSFORMERS_CONFIG_WARNING + " Extra context.")
                transformers_logger.warning("unrelated transformers warning")
                warnings.warn_explicit(
                    "Could not load WhisperProcessor: missing preprocessor_config.json.",
                    UserWarning,
                    filename="whisper.py",
                    lineno=1,
                    module="mlx_audio.stt.models.whisper.whisper",
                )
                warnings.warn("unrelated user warning", UserWarning)

    assert server._QWEN_TRANSFORMERS_CONFIG_WARNING not in caplog.text
    assert "unrelated transformers warning" in caplog.text
    assert [str(warning.message) for warning in caught] == ["unrelated user warning"]


def test_huggingface_loader_noise_defaults_are_quiet() -> None:
    import agent_voice.server  # noqa: F401

    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"


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
            assert "verbose" not in kwargs
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

from __future__ import annotations

from pathlib import Path

from tests.helpers.public_contract import MockSpeechServer, require_executable, run_with_home


def test_agent_speak_no_args_is_safe_success(tmp_path: Path) -> None:
    agent_speak = require_executable("agent-speak")

    result = run_with_home([str(agent_speak)], tmp_path)

    assert result.returncode == 0, result.stderr


def test_codex_speak_alias_no_args_is_safe_success(tmp_path: Path) -> None:
    codex_speak = require_executable("codex-speak")

    result = run_with_home([str(codex_speak)], tmp_path)

    assert result.returncode == 0, result.stderr


def test_agent_voice_summary_rejects_empty_input(tmp_path: Path) -> None:
    helper = require_executable("agent-voice-summary")

    result = run_with_home([str(helper), "--no-play"], tmp_path, input_text="   \n\t")

    assert result.returncode == 2


def test_agent_voice_summary_calls_mock_server_and_writes_output(tmp_path: Path) -> None:
    helper = require_executable("agent-voice-summary")
    output = tmp_path / "summary.wav"

    with MockSpeechServer() as server:
        result = run_with_home(
            [
                str(helper),
                "--server",
                server.url,
                "--voice",
                "cyberpunk_cool",
                "--output",
                str(output),
                "--no-play",
                "  hello from tests  ",
            ],
            tmp_path,
        )

    assert result.returncode == 0, result.stderr
    assert output.read_bytes().startswith(b"RIFF")
    assert len(server.requests) == 1
    request = server.requests[0].body
    assert request["input"] == "hello from tests"
    assert request["voice"] == "cyberpunk_cool"
    assert request.get("model", "qwen3-tts") == "qwen3-tts"


def test_agent_voice_summary_sends_max_tokens_when_requested(tmp_path: Path) -> None:
    helper = require_executable("agent-voice-summary")

    with MockSpeechServer() as server:
        result = run_with_home(
            [
                str(helper),
                "--server",
                server.url,
                "--max-tokens",
                "32123",
                "--no-play",
                "hello from tests",
            ],
            tmp_path,
        )

    assert result.returncode == 0, result.stderr
    assert server.requests[0].body["max_tokens"] == 32123


def test_agent_voice_summary_rejects_invalid_max_tokens(tmp_path: Path) -> None:
    helper = require_executable("agent-voice-summary")

    result = run_with_home([str(helper), "--max-tokens", "abc", "--no-play", "hello"], tmp_path)

    assert result.returncode == 2
    assert "--max-tokens" in result.stderr


def test_agent_voice_summary_rejects_voice_not_advertised_by_server(tmp_path: Path) -> None:
    helper = require_executable("agent-voice-summary")

    with MockSpeechServer(voices=["warm_wisdom"]) as server:
        result = run_with_home(
            [str(helper), "--server", server.url, "--voice", "cyberpunk_cool", "--no-play", "hello"],
            tmp_path,
        )

    assert result.returncode != 0
    assert server.requests == []


def test_agent_voice_summary_allows_custom_instruct_without_listed_voice(tmp_path: Path) -> None:
    helper = require_executable("agent-voice-summary")
    output = tmp_path / "summary.wav"

    with MockSpeechServer(voices=["warm_wisdom"]) as server:
        result = run_with_home(
            [
                str(helper),
                "--server",
                server.url,
                "--voice",
                "custom_contract_voice",
                "--instruct",
                "Speak warmly and clearly.",
                "--output",
                str(output),
                "--no-play",
                "hello",
            ],
            tmp_path,
        )

    assert result.returncode == 0, result.stderr
    assert len(server.requests) == 1
    request = server.requests[0].body
    assert request["voice"] == "custom_contract_voice"
    assert request["instruct"] == "Speak warmly and clearly."


def test_codex_voice_summary_alias_still_works(tmp_path: Path) -> None:
    helper = require_executable("codex-voice-summary")
    output = tmp_path / "summary.wav"

    with MockSpeechServer() as server:
        result = run_with_home(
            [
                str(helper),
                "--server",
                server.url,
                "--output",
                str(output),
                "--no-play",
                "hello from alias",
            ],
            tmp_path,
        )

    assert result.returncode == 0, result.stderr
    assert output.read_bytes().startswith(b"RIFF")
    assert server.requests[0].body["input"] == "hello from alias"

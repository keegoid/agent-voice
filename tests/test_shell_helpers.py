from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path

from tests.helpers.public_contract import MockSpeechServer, make_fake_bin, require_executable, run_with_home


def test_agent_speak_no_args_is_safe_success(tmp_path: Path) -> None:
    agent_speak = require_executable("agent-speak")

    result = run_with_home([str(agent_speak)], tmp_path)

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


def test_agent_voice_summary_skips_speech_when_server_muted(tmp_path: Path) -> None:
    helper = require_executable("agent-voice-summary")

    with MockSpeechServer(muted=True) as server:
        result = run_with_home(
            [str(helper), "--server", server.url, "--no-play", "hello from tests"],
            tmp_path,
        )

    assert result.returncode == 0, result.stderr
    assert server.requests == []


def test_agent_voice_mute_commands_update_state(tmp_path: Path) -> None:
    command = require_executable("agent-voice")
    state = tmp_path / "mute.json"
    env = {"AGENT_VOICE_MUTE_STATE": str(state)}

    muted = run_with_home([str(command), "mute"], tmp_path, extra_env=env)
    status = run_with_home([str(command), "mute", "status"], tmp_path, extra_env=env)
    unmuted = run_with_home([str(command), "unmute"], tmp_path, extra_env=env)

    assert muted.returncode == 0, muted.stderr
    assert muted.stdout.strip() == "muted"
    assert status.stdout.strip() == "muted"
    assert unmuted.returncode == 0, unmuted.stderr
    assert unmuted.stdout.strip() == "unmuted"
    assert json.loads(state.read_text(encoding="utf-8"))["muted"] is False


def test_agent_voice_mute_command_rejects_mutation_with_env_override(tmp_path: Path) -> None:
    command = require_executable("agent-voice")
    state = tmp_path / "mute.json"
    env = {"AGENT_VOICE_MUTE_STATE": str(state), "AGENT_VOICE_MUTED": "true"}

    status = run_with_home([str(command), "mute", "status"], tmp_path, extra_env=env)
    changed = run_with_home([str(command), "mute", "off"], tmp_path, extra_env=env)

    assert status.returncode == 0, status.stderr
    assert status.stdout.strip() == "muted"
    assert changed.returncode == 1
    assert "AGENT_VOICE_MUTED" in changed.stderr
    assert not state.exists()


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


def test_agent_voice_summary_times_out_hung_afplay(tmp_path: Path) -> None:
    helper = require_executable("agent-voice-summary")
    fake_bin = tmp_path / "home" / ".local" / "bin"
    make_fake_bin(
        fake_bin,
        "afplay",
        """
        #!/usr/bin/env bash
        sleep 5
        """,
    )

    with MockSpeechServer() as server:
        result = run_with_home(
            [
                str(helper),
                "--server",
                server.url,
                "--play-timeout",
                "0.2",
                "hello from tests",
            ],
            tmp_path,
            timeout=5,
        )

    assert result.returncode == 124
    assert "afplay timed out" in result.stderr


def test_agent_voice_summary_uses_wav_suffix_for_temp_playback(tmp_path: Path) -> None:
    helper = require_executable("agent-voice-summary")
    fake_bin = tmp_path / "home" / ".local" / "bin"
    seen_path = tmp_path / "played-path.txt"
    make_fake_bin(
        fake_bin,
        "afplay",
        f"""
        #!/usr/bin/env bash
        printf '%s' "$1" > "{seen_path}"
        """,
    )

    with MockSpeechServer() as server:
        result = run_with_home(
            [
                str(helper),
                "--server",
                server.url,
                "hello from tests",
            ],
            tmp_path,
        )

    assert result.returncode == 0, result.stderr
    assert seen_path.read_text(encoding="utf-8").endswith(".wav")


def test_agent_voice_summary_rejects_invalid_playback_timeout(tmp_path: Path) -> None:
    helper = require_executable("agent-voice-summary")

    result = run_with_home([str(helper), "--play-timeout", "abc", "hello"], tmp_path)

    assert result.returncode == 2
    assert "Playback timeout" in result.stderr


def _make_git_source_with_installer(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    installer = source / "install.sh"
    installer.write_text(
        textwrap.dedent(
            """
            #!/usr/bin/env bash
            set -euo pipefail
            mkdir -p "$AGENT_VOICE_HOME"
            printf '%s\n' "$@" > "$AGENT_VOICE_HOME/install-args.txt"
            """
        ).lstrip(),
        encoding="utf-8",
    )
    installer.chmod(0o755)
    subprocess.run(["git", "init"], cwd=source, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=source, check=True)
    subprocess.run(["git", "add", "install.sh"], cwd=source, check=True)
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "commit", "-m", "initial"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    )
    return source


def test_sync_installed_records_manifest_and_uses_noninteractive_install(tmp_path: Path) -> None:
    helper = require_executable("sync-installed")
    source = _make_git_source_with_installer(tmp_path)
    state = tmp_path / "state"

    result = subprocess.run(
        [str(helper), "--source-dir", str(source), "--state-dir", str(state), "--no-verify"],
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert "--no-codex-config" in (state / "install-args.txt").read_text(encoding="utf-8")
    manifest = json.loads((state / "install-manifest.json").read_text(encoding="utf-8"))
    assert manifest["repo"] == str(source)
    assert manifest["commit"] == subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    assert manifest["dirty"] is False


def test_sync_installed_refuses_dirty_source_without_override(tmp_path: Path) -> None:
    helper = require_executable("sync-installed")
    source = _make_git_source_with_installer(tmp_path)
    (source / "install.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")

    result = subprocess.run(
        [str(helper), "--source-dir", str(source), "--state-dir", str(tmp_path / "state"), "--no-verify"],
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 1
    assert "source tree is dirty" in result.stderr

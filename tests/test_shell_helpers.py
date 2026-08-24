from __future__ import annotations

import json
import os
import subprocess
import textwrap
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from tests.helpers.public_contract import MockSpeechServer, make_fake_bin, require_executable, run_with_home


def test_agent_speak_no_args_is_safe_success(tmp_path: Path) -> None:
    agent_speak = require_executable("agent-speak")

    result = run_with_home([str(agent_speak)], tmp_path)

    assert result.returncode == 0, result.stderr


def test_agent_speak_prepends_configured_speaker_label(tmp_path: Path) -> None:
    agent_speak = require_executable("agent-speak")
    capture = tmp_path / "spoken.txt"
    helper = make_fake_bin(
        tmp_path / "bin",
        "fake-voice-helper",
        f"""
        #!/usr/bin/env bash
        printf '%s\\n' "$*" >> "{capture}"
        """,
    )

    result = run_with_home(
        [str(agent_speak), "hello from tests"],
        tmp_path,
        extra_env={
            "AGENT_VOICE_HELPER": str(helper),
            "AGENT_VOICE_SPEAK_SYNC": "1",
            "AGENT_VOICE_SPEAK_LOCK": str(tmp_path / "speak.lock"),
            "AGENT_VOICE_SPEAKER_LABEL": " DEV CEO ",
        },
    )

    assert result.returncode == 0, result.stderr
    assert capture.read_text(encoding="utf-8").strip() == "DEV CEO. hello from tests"


def test_agent_speak_does_not_duplicate_existing_label(tmp_path: Path) -> None:
    agent_speak = require_executable("agent-speak")
    capture = tmp_path / "spoken.txt"
    helper = make_fake_bin(
        tmp_path / "bin",
        "fake-voice-helper",
        f"""
        #!/usr/bin/env bash
        printf '%s\\n' "$*" >> "{capture}"
        """,
    )

    result = run_with_home(
        [str(agent_speak), "DEV CEO. already labeled"],
        tmp_path,
        extra_env={
            "AGENT_VOICE_HELPER": str(helper),
            "AGENT_VOICE_SPEAK_SYNC": "1",
            "AGENT_VOICE_SPEAK_LOCK": str(tmp_path / "speak.lock"),
            "AGENT_VOICE_SPEAKER_LABEL": "DEV CEO",
        },
    )

    assert result.returncode == 0, result.stderr
    assert capture.read_text(encoding="utf-8").strip() == "DEV CEO. already labeled"


def test_agent_speak_skips_when_voice_lock_is_busy(tmp_path: Path) -> None:
    agent_speak = require_executable("agent-speak")
    capture = tmp_path / "spoken.txt"
    log = tmp_path / "agent-speak.log"
    lock = tmp_path / "speak.lock"
    lock.mkdir()
    (lock / "pid").write_text(str(os.getpid()), encoding="utf-8")
    helper = make_fake_bin(
        tmp_path / "bin",
        "fake-voice-helper",
        f"""
        #!/usr/bin/env bash
        printf '%s\\n' "$*" >> "{capture}"
        """,
    )

    result = run_with_home(
        [str(agent_speak), "hello from tests"],
        tmp_path,
        extra_env={
            "AGENT_VOICE_HELPER": str(helper),
            "AGENT_VOICE_SPEAK_SYNC": "1",
            "AGENT_VOICE_SPEAK_LOCK": str(lock),
            "AGENT_VOICE_SPEAK_LOCK_WAIT_SECONDS": "0",
            "AGENT_VOICE_SPEAK_LOG": str(log),
        },
    )

    assert result.returncode == 0, result.stderr
    assert not capture.exists()
    assert "voice lock busy" in log.read_text(encoding="utf-8")


def test_agent_speak_spools_when_helper_cannot_reach_server(tmp_path: Path) -> None:
    agent_speak = require_executable("agent-speak")
    spool = tmp_path / "spool"
    helper = make_fake_bin(
        tmp_path / "bin",
        "fake-voice-helper",
        """
        #!/usr/bin/env bash
        exit 7
        """,
    )

    result = run_with_home(
        [str(agent_speak), "--voice", "warm_wisdom", "hello from sandbox"],
        tmp_path,
        extra_env={
            "AGENT_VOICE_HELPER": str(helper),
            "AGENT_VOICE_SPEAK_SYNC": "1",
            "AGENT_VOICE_SPEAK_LOCK": str(tmp_path / "speak.lock"),
            "AGENT_VOICE_SPOOL_DIR": str(spool),
        },
    )

    assert result.returncode == 0, result.stderr
    queued = list(spool.glob("*.json"))
    assert len(queued) == 1
    payload = json.loads(queued[0].read_text(encoding="utf-8"))
    assert payload["message"] == "hello from sandbox"
    assert payload["voice_id"] == "warm_wisdom"


def test_agent_speak_spool_temp_template_is_portable(tmp_path: Path) -> None:
    agent_speak = require_executable("agent-speak")
    spool = tmp_path / "spool"
    mktemp_template = tmp_path / "mktemp-template.txt"
    home_bin = tmp_path / "home" / ".local" / "bin"
    make_fake_bin(
        home_bin,
        "mktemp",
        f"""
        #!/usr/bin/env bash
        printf '%s\\n' "$1" > "{mktemp_template}"
        case "$1" in
          *XXXXXX)
            path="${{1%XXXXXX}}abc123"
            : > "$path"
            printf '%s\\n' "$path"
            ;;
          *)
            echo "mktemp template must end with XXXXXX" >&2
            exit 64
            ;;
        esac
        """,
    )
    helper = make_fake_bin(
        tmp_path / "bin",
        "fake-voice-helper",
        """
        #!/usr/bin/env bash
        exit 7
        """,
    )

    result = run_with_home(
        [str(agent_speak), "hello from portable spool"],
        tmp_path,
        extra_env={
            "AGENT_VOICE_HELPER": str(helper),
            "AGENT_VOICE_SPEAK_SYNC": "1",
            "AGENT_VOICE_SPEAK_LOCK": str(tmp_path / "speak.lock"),
            "AGENT_VOICE_SPOOL_DIR": str(spool),
        },
    )

    assert result.returncode == 0, result.stderr
    assert mktemp_template.read_text(encoding="utf-8").strip().endswith(".agent-speak.XXXXXX")
    queued = list(spool.glob("*.json"))
    assert len(queued) == 1
    payload = json.loads(queued[0].read_text(encoding="utf-8"))
    assert payload["message"] == "hello from portable spool"


class MockPaperclipServer:
    def __init__(self) -> None:
        self.requests: list[str] = []
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        assert self._server is not None
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    def __enter__(self) -> "MockPaperclipServer":
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                outer.requests.append(self.path)
                if self.path == "/api/agents/me":
                    payload = {
                        "id": "agent-1",
                        "companyId": "company-1",
                        "name": "keegoid-codex",
                        "role": "reviewer",
                        "title": "Codex PR Owner",
                    }
                elif self.path == "/api/companies/company-1":
                    payload = {"id": "company-1", "name": "DEVELOPMENT", "issuePrefix": "DEV"}
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode())

            def log_message(self, *_args: object) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if self._thread:
            self._thread.join(timeout=2)


def test_agent_speak_resolves_paperclip_speaker_label_from_api(tmp_path: Path) -> None:
    agent_speak = require_executable("agent-speak")
    capture = tmp_path / "spoken.txt"
    helper = make_fake_bin(
        tmp_path / "bin",
        "fake-voice-helper",
        f"""
        #!/usr/bin/env bash
        printf '%s\\n' "$*" >> "{capture}"
        """,
    )

    with MockPaperclipServer() as server:
        result = run_with_home(
            [str(agent_speak), "opening the PR"],
            tmp_path,
            extra_env={
                "AGENT_VOICE_HELPER": str(helper),
                "AGENT_VOICE_SPEAK_SYNC": "1",
                "AGENT_VOICE_SPEAK_LOCK": str(tmp_path / "speak.lock"),
                "PAPERCLIP_AGENT_ID": "agent-1",
                "PAPERCLIP_COMPANY_ID": "company-1",
                "PAPERCLIP_API_URL": server.url,
                "PAPERCLIP_API_KEY": "test-token",
            },
        )

    assert result.returncode == 0, result.stderr
    assert capture.read_text(encoding="utf-8").strip() == "DEV keegoid-codex. opening the PR"
    assert "/api/agents/me" in server.requests
    assert "/api/companies/company-1" in server.requests


def test_agent_speak_does_not_send_token_to_untrusted_paperclip_api_url(tmp_path: Path) -> None:
    agent_speak = require_executable("agent-speak")
    capture = tmp_path / "spoken.txt"
    curl_log = tmp_path / "curl.log"
    fake_bin = tmp_path / "bin"
    helper = make_fake_bin(
        fake_bin,
        "fake-voice-helper",
        f"""
        #!/usr/bin/env bash
        printf '%s\\n' "$*" >> "{capture}"
        """,
    )
    make_fake_bin(
        fake_bin,
        "curl",
        f"""
        #!/usr/bin/env bash
        printf '%s\\n' "$*" >> "{curl_log}"
        exit 99
        """,
    )

    result = run_with_home(
        [str(agent_speak), "opening the PR"],
        tmp_path,
        extra_env={
            "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
            "AGENT_VOICE_HELPER": str(helper),
            "AGENT_VOICE_SPEAK_SYNC": "1",
            "AGENT_VOICE_SPEAK_LOCK": str(tmp_path / "speak.lock"),
            "PAPERCLIP_AGENT_ID": "agent-1",
            "PAPERCLIP_COMPANY_ID": "company-1",
            "PAPERCLIP_API_URL": "https://attacker.example",
            "PAPERCLIP_API_KEY": "secret-token",
        },
    )

    assert result.returncode == 0, result.stderr
    assert capture.read_text(encoding="utf-8").strip() == "opening the PR"
    assert not curl_log.exists()


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


def test_agent_voice_summary_defaults_to_chesapeake_balanced(tmp_path: Path) -> None:
    helper = require_executable("agent-voice-summary")
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
                "hello from tests",
            ],
            tmp_path,
        )

    assert result.returncode == 0, result.stderr
    assert len(server.requests) == 1
    assert server.requests[0].body["voice"] == "chesapeake_balanced"


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


def _wav_bytes(tmp_path: Path, duration_seconds: float, *, sample_rate: int = 24_000) -> bytes:
    wav_path = tmp_path / "agent-voice-test.wav"
    with wave.open(str(wav_path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(b"\0\0" * int(duration_seconds * sample_rate))
    data = wav_path.read_bytes()
    wav_path.unlink(missing_ok=True)
    return data


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


def test_agent_voice_summary_refuses_to_play_wav_over_three_minutes(tmp_path: Path) -> None:
    helper = require_executable("agent-voice-summary")
    fake_bin = tmp_path / "home" / ".local" / "bin"
    played = tmp_path / "played.txt"
    make_fake_bin(
        fake_bin,
        "afplay",
        f"""
        #!/usr/bin/env bash
        printf played > "{played}"
        """,
    )

    with MockSpeechServer(audio=_wav_bytes(tmp_path, 181)) as server:
        result = run_with_home(
            [str(helper), "--server", server.url, "short voice cue"],
            tmp_path,
        )

    assert result.returncode == 64
    assert "Refusing suspiciously long TTS output" in result.stderr
    assert not played.exists()


def test_agent_voice_summary_allows_three_minute_playback_cap_override(tmp_path: Path) -> None:
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

    with MockSpeechServer(audio=_wav_bytes(tmp_path, 181)) as server:
        result = run_with_home(
            [str(helper), "--server", server.url, "--max-playback-seconds", "240", "short voice cue"],
            tmp_path,
        )

    assert result.returncode == 0, result.stderr
    assert seen_path.read_text(encoding="utf-8").endswith(".wav")


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

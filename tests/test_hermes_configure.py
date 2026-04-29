from __future__ import annotations

import io
from pathlib import Path

from ruamel.yaml import YAML

from tests.helpers.public_contract import require_executable, run_with_home, tree_snapshot


def _load_yaml(text: str):
    return YAML().load(io.StringIO(text))


def test_configure_hermes_dry_run_does_not_modify_files(tmp_path: Path) -> None:
    command = require_executable("agent-voice")
    home = tmp_path / "home"
    hermes = home / ".hermes"
    hermes.mkdir(parents=True)
    (hermes / "config.yaml").write_text(
        "tts:\n  provider: edge\nvoice:\n  auto_tts: false\n",
        encoding="utf-8",
    )
    (hermes / ".env").write_text("# existing env\n", encoding="utf-8")
    before = tree_snapshot(home)

    result = run_with_home([str(command), "configure", "hermes", "--dry-run"], tmp_path)

    assert result.returncode == 0, result.stderr
    assert "dry-run" in result.stdout
    assert tree_snapshot(home) == before


def test_configure_hermes_updates_config_env_and_writes_backups(tmp_path: Path) -> None:
    command = require_executable("agent-voice")
    home = tmp_path / "home"
    hermes = home / ".hermes"
    hermes.mkdir(parents=True)
    config = hermes / "config.yaml"
    env = hermes / ".env"
    config.write_text(
        "\n".join(
            [
                "model:",
                "  provider: openai-codex",
                "tts:",
                "  provider: edge",
                "  edge:",
                "    voice: en-US-AriaNeural",
                "  openai:",
                "    model: gpt-4o-mini-tts",
                "    voice: alloy",
                "voice:",
                "  auto_tts: false",
                "",
            ]
        ),
        encoding="utf-8",
    )
    env.write_text("# existing env\nVOICE_TOOLS_OPENAI_KEY=old-value\n", encoding="utf-8")

    result = run_with_home(
        [
            str(command),
            "configure",
            "hermes",
            "--server",
            "http://127.0.0.1:9999/v1",
            "--voice",
            "warm_wisdom",
        ],
        tmp_path,
    )

    assert result.returncode == 0, result.stderr
    updated_config = config.read_text(encoding="utf-8")
    updated_env = env.read_text(encoding="utf-8")
    loaded = _load_yaml(updated_config)
    assert loaded["tts"]["provider"] == "openai"
    assert loaded["tts"]["openai"]["model"] == "qwen3-tts"
    assert loaded["tts"]["openai"]["voice"] == "warm_wisdom"
    assert loaded["tts"]["openai"]["base_url"] == "http://127.0.0.1:9999/v1"
    assert loaded["voice"]["auto_tts"] is True
    assert "VOICE_TOOLS_OPENAI_KEY=old-value" in updated_env

    backups = sorted((home / ".agent-voice" / "backups").glob("*-hermes-*"))
    assert backups
    assert any(path.name == "config.yaml" for path in backups[0].rglob("*"))
    assert any(path.name == ".env" for path in backups[0].rglob("*"))


def test_configure_hermes_creates_missing_blocks(tmp_path: Path) -> None:
    command = require_executable("agent-voice")
    home = tmp_path / "home"
    hermes = home / ".hermes"
    hermes.mkdir(parents=True)
    (hermes / "config.yaml").write_text("model:\n  provider: openai-codex\n", encoding="utf-8")

    result = run_with_home([str(command), "configure", "hermes"], tmp_path)

    assert result.returncode == 0, result.stderr
    updated_config = (hermes / "config.yaml").read_text(encoding="utf-8")
    updated_env = (hermes / ".env").read_text(encoding="utf-8")
    assert "tts:" in updated_config
    assert "  provider: openai" in updated_config
    assert "  openai:" in updated_config
    assert "voice:" in updated_config
    assert "  auto_tts: true" in updated_config
    assert "VOICE_TOOLS_OPENAI_KEY=agent-voice-local" in updated_env


def test_configure_hermes_handles_inline_maps_comments_and_wide_indents(tmp_path: Path) -> None:
    command = require_executable("agent-voice")
    home = tmp_path / "home"
    hermes = home / ".hermes"
    hermes.mkdir(parents=True)
    config = hermes / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "# keep top comment",
                "tts:",
                "    provider: edge",
                "    openai: {model: old-model, voice: alloy} # provider block",
                "voice: {auto_tts: false} # voice block",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = run_with_home([str(command), "configure", "hermes"], tmp_path)

    assert result.returncode == 0, result.stderr
    updated_config = config.read_text(encoding="utf-8")
    loaded = _load_yaml(updated_config)
    assert sum(1 for line in updated_config.splitlines() if line == "tts:") == 1
    assert updated_config.startswith("# keep top comment")
    assert "provider block" in updated_config
    assert "voice block" in updated_config
    assert loaded["tts"]["provider"] == "openai"
    assert loaded["tts"]["openai"]["model"] == "qwen3-tts"
    assert loaded["tts"]["openai"]["voice"] == "cyberpunk_cool"
    assert loaded["tts"]["openai"]["base_url"] == "http://127.0.0.1:8880/v1"
    assert loaded["voice"]["auto_tts"] is True


def test_configure_hermes_sets_empty_voice_tools_key(tmp_path: Path) -> None:
    command = require_executable("agent-voice")
    home = tmp_path / "home"
    hermes = home / ".hermes"
    hermes.mkdir(parents=True)
    (hermes / "config.yaml").write_text("tts: {}\n", encoding="utf-8")
    (hermes / ".env").write_text("VOICE_TOOLS_OPENAI_KEY=\n", encoding="utf-8")

    result = run_with_home([str(command), "configure", "hermes"], tmp_path)

    assert result.returncode == 0, result.stderr
    assert (hermes / ".env").read_text(encoding="utf-8") == "VOICE_TOOLS_OPENAI_KEY=agent-voice-local\n"


def test_configure_hermes_rejects_outside_home_without_opt_in(tmp_path: Path) -> None:
    command = require_executable("agent-voice")
    outside = tmp_path / "outside-hermes"

    result = run_with_home(
        [str(command), "configure", "hermes", "--hermes-home", str(outside)],
        tmp_path,
    )

    assert result.returncode == 2
    assert "outside HOME" in result.stderr
    assert not outside.exists()


def test_configure_hermes_allows_outside_home_with_opt_in(tmp_path: Path) -> None:
    command = require_executable("agent-voice")
    outside = tmp_path / "outside-hermes"

    result = run_with_home(
        [
            str(command),
            "configure",
            "hermes",
            "--hermes-home",
            str(outside),
            "--allow-outside-home",
        ],
        tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert (outside / "config.yaml").exists()

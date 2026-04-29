from __future__ import annotations

from pathlib import Path

from tests.helpers.public_contract import require_executable, run_with_home, tree_snapshot


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
    assert "  provider: openai" in updated_config
    assert "    model: 'qwen3-tts'" in updated_config
    assert "    voice: 'warm_wisdom'" in updated_config
    assert "    base_url: 'http://127.0.0.1:9999/v1'" in updated_config
    assert "  auto_tts: true" in updated_config
    assert "VOICE_TOOLS_OPENAI_KEY=agent-voice-local" in updated_env

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

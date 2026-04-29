"""Configure Hermes Agent to use the local agent-voice TTS server."""

from __future__ import annotations

import argparse
import io
import os
import shutil
import subprocess
import sys
import time
from collections.abc import MutableMapping
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap


DEFAULT_SERVER_URL = "http://127.0.0.1:8880/v1"
DEFAULT_VOICE = "cyberpunk_cool"
LOCAL_AUDIO_KEY = "agent-voice-local"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent-voice configure hermes",
        description="Configure Hermes Agent to use agent-voice for TTS.",
    )
    parser.add_argument(
        "--hermes-home",
        default=os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes"),
        help="Hermes home directory. Default: ~/.hermes or HERMES_HOME.",
    )
    parser.add_argument(
        "--server",
        default=os.environ.get("AGENT_VOICE_SERVER") or DEFAULT_SERVER_URL,
        help=f"agent-voice OpenAI-compatible base URL. Default: {DEFAULT_SERVER_URL}.",
    )
    parser.add_argument(
        "--voice",
        default=os.environ.get("AGENT_VOICE_VOICE") or DEFAULT_VOICE,
        help=f"agent-voice preset voice. Default: {DEFAULT_VOICE}.",
    )
    parser.add_argument(
        "--auto-tts",
        choices=("true", "false"),
        default="true",
        help="Set Hermes voice.auto_tts. Default: true.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes without writing files.",
    )
    parser.add_argument(
        "--restart-gateway",
        action="store_true",
        help="Run 'hermes gateway restart' after writing config.",
    )
    parser.add_argument(
        "--allow-outside-home",
        action="store_true",
        help="Allow --hermes-home outside the current HOME.",
    )
    args = parser.parse_args(argv)

    hermes_home = Path(args.hermes_home).expanduser()
    if not args.allow_outside_home and not _is_within_home(hermes_home):
        print(
            f"Refusing to edit Hermes home outside HOME without --allow-outside-home: {hermes_home}",
            file=sys.stderr,
        )
        return 2
    config_path = hermes_home / "config.yaml"
    env_path = hermes_home / ".env"
    agent_voice_home = Path(os.environ.get("AGENT_VOICE_HOME") or Path.home() / ".agent-voice")
    backup_root = agent_voice_home / "backups" / f"{time.strftime('%Y%m%d%H%M%S')}-hermes-{os.getpid()}"

    planned = [
        f"{config_path}: tts.provider=openai",
        f"{config_path}: tts.openai.model=qwen3-tts",
        f"{config_path}: tts.openai.voice={args.voice}",
        f"{config_path}: tts.openai.base_url={args.server.rstrip('/')}",
        f"{config_path}: voice.auto_tts={args.auto_tts}",
        f"{env_path}: VOICE_TOOLS_OPENAI_KEY={LOCAL_AUDIO_KEY}",
    ]

    if args.dry_run:
        print("agent-voice configure hermes dry-run:")
        for item in planned:
            print(f"  would set {item}")
        if args.restart_gateway:
            print("  would run hermes gateway restart")
        return 0

    hermes_home.mkdir(parents=True, exist_ok=True)
    backup_file(config_path, backup_root)
    backup_file(env_path, backup_root)

    try:
        config_text = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
        config_text = configure_yaml(
            config_text,
            server_url=args.server.rstrip("/"),
            voice=args.voice,
            auto_tts=args.auto_tts,
        )
    except Exception as exc:
        print(f"Failed to update {config_path}: {exc}", file=sys.stderr)
        return 1
    config_path.write_text(config_text, encoding="utf-8")

    env_text = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    env_path.write_text(configure_env(env_text), encoding="utf-8")

    print("Configured Hermes for agent-voice TTS.")
    print(f"  Config: {config_path}")
    print(f"  Env:    {env_path}")
    print(f"  Backup: {backup_root}")

    if args.restart_gateway:
        restart = subprocess.run(["hermes", "gateway", "restart"], text=True)
        if restart.returncode != 0:
            print("hermes gateway restart failed", file=sys.stderr)
            return restart.returncode

    return 0


def backup_file(path: Path, backup_root: Path) -> None:
    if not path.exists():
        return
    home = Path.home()
    try:
        rel = Path("home") / path.relative_to(home)
    except ValueError:
        rel = Path("external") / Path(*path.resolve().parts[1:])
    destination = backup_root / rel
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)


def configure_yaml(text: str, *, server_url: str, voice: str, auto_tts: str) -> str:
    yaml = _yaml()
    data = yaml.load(text) if text.strip() else CommentedMap()
    if data is None:
        data = CommentedMap()
    if not isinstance(data, MutableMapping):
        raise ValueError("config.yaml root must be a YAML mapping")

    tts = _ensure_mapping(data, "tts")
    tts["provider"] = "openai"
    openai = _ensure_mapping(tts, "openai")
    openai["model"] = "qwen3-tts"
    openai["voice"] = voice
    openai["base_url"] = server_url

    voice_config = _ensure_mapping(data, "voice")
    voice_config["auto_tts"] = auto_tts == "true"

    output = io.StringIO()
    yaml.dump(data, output)
    return output.getvalue()


def configure_env(text: str) -> str:
    lines = text.splitlines()
    replacement = f"VOICE_TOOLS_OPENAI_KEY={LOCAL_AUDIO_KEY}"
    for index, line in enumerate(lines):
        if line.startswith("VOICE_TOOLS_OPENAI_KEY="):
            _, _, value = line.partition("=")
            if not value.strip():
                lines[index] = replacement
            return "\n".join(lines).rstrip() + "\n"
    if lines and lines[-1].strip():
        lines.append("")
    lines.append("# Local OpenAI-compatible audio key used by agent-voice.")
    lines.append(replacement)
    return "\n".join(lines).rstrip() + "\n"


def _yaml() -> YAML:
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.allow_duplicate_keys = False
    yaml.default_flow_style = False
    return yaml


def _ensure_mapping(parent: MutableMapping, key: str) -> CommentedMap:
    value = parent.get(key)
    if isinstance(value, MutableMapping):
        return value
    replacement = CommentedMap()
    parent[key] = replacement
    return replacement


def _is_within_home(path: Path) -> bool:
    try:
        path.resolve().relative_to(Path.home().resolve())
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())

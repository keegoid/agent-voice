"""Configure Hermes Agent to use the local agent-voice TTS server."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


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
    args = parser.parse_args(argv)

    hermes_home = Path(args.hermes_home).expanduser()
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

    config_text = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    config_text = configure_yaml(
        config_text,
        server_url=args.server.rstrip("/"),
        voice=args.voice,
        auto_tts=args.auto_tts,
    )
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
    lines = text.splitlines()
    if not lines:
        lines = []

    lines = set_top_child(lines, "tts", "provider", "openai")
    lines = set_nested_child(lines, "tts", "openai", "model", quote_yaml("qwen3-tts"))
    lines = set_nested_child(lines, "tts", "openai", "voice", quote_yaml(voice))
    lines = set_nested_child(lines, "tts", "openai", "base_url", quote_yaml(server_url))
    lines = set_top_child(lines, "voice", "auto_tts", auto_tts)
    return "\n".join(lines).rstrip() + "\n"


def configure_env(text: str) -> str:
    lines = text.splitlines()
    replacement = f"VOICE_TOOLS_OPENAI_KEY={LOCAL_AUDIO_KEY}"
    for index, line in enumerate(lines):
        if line.startswith("VOICE_TOOLS_OPENAI_KEY="):
            lines[index] = replacement
            return "\n".join(lines).rstrip() + "\n"
    if lines and lines[-1].strip():
        lines.append("")
    lines.append("# Local OpenAI-compatible audio key used by agent-voice.")
    lines.append(replacement)
    return "\n".join(lines).rstrip() + "\n"


def quote_yaml(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def set_top_child(lines: list[str], block_key: str, child_key: str, value: str) -> list[str]:
    start, end = ensure_top_block(lines, block_key)
    target_prefix = f"  {child_key}:"
    replacement = f"  {child_key}: {value}"
    for index in range(start + 1, end):
        if lines[index].startswith(target_prefix):
            lines[index] = replacement
            return lines
    lines.insert(start + 1, replacement)
    return lines


def set_nested_child(
    lines: list[str],
    block_key: str,
    nested_key: str,
    child_key: str,
    value: str,
) -> list[str]:
    nested_start = ensure_nested_block(lines, block_key, nested_key)
    nested_end = nested_block_end(lines, nested_start)
    target_prefix = f"    {child_key}:"
    replacement = f"    {child_key}: {value}"
    for index in range(nested_start + 1, nested_end):
        if lines[index].startswith(target_prefix):
            lines[index] = replacement
            return lines
    lines.insert(nested_end, replacement)
    return lines


def ensure_top_block(lines: list[str], block_key: str) -> tuple[int, int]:
    found = find_top_block(lines, block_key)
    if found:
        return found
    if lines and lines[-1].strip():
        lines.append("")
    start = len(lines)
    lines.append(f"{block_key}:")
    return start, len(lines)


def ensure_nested_block(lines: list[str], block_key: str, nested_key: str) -> int:
    start, end = ensure_top_block(lines, block_key)
    target_prefix = f"  {nested_key}:"
    for index in range(start + 1, end):
        if lines[index].startswith(target_prefix):
            lines[index] = f"  {nested_key}:"
            return index
    lines.insert(end, f"  {nested_key}:")
    return end


def find_top_block(lines: list[str], block_key: str) -> tuple[int, int] | None:
    prefix = f"{block_key}:"
    for index, line in enumerate(lines):
        if line == prefix or line.startswith(f"{prefix} "):
            return index, top_block_end(lines, index)
    return None


def top_block_end(lines: list[str], start: int) -> int:
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if line and not line.startswith((" ", "#")):
            return index
    return len(lines)


def nested_block_end(lines: list[str], start: int) -> int:
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent <= 2:
            return index
    return len(lines)


if __name__ == "__main__":
    raise SystemExit(main())

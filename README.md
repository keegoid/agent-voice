# codex-tts

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-black.svg)](#requirements)
[![Python](https://img.shields.io/badge/python-3.12%2B-3776AB.svg)](pyproject.toml)
[![API](https://img.shields.io/badge/API-OpenAI%20speech%20subset-009688.svg)](#api)

Local TTS for Codex progress cues on macOS Apple Silicon. It serves a small
OpenAI-compatible speech endpoint backed by Qwen3-TTS through MLX, plus shell
helpers that make voice cues best-effort instead of task-breaking.

## Install

Version 1 supports macOS Apple Silicon only.

```bash
tmp="$(mktemp -d)" && curl -fsSL https://raw.githubusercontent.com/keegoid/codex-tts/main/install.sh -o "$tmp/install.sh" && bash "$tmp/install.sh"
```

The installer downloads a temporary installer first, then runs it. It does not
pipe remote code into a shell.

## Requirements

- macOS on Apple Silicon.
- Python 3.12 or newer.
- `uv`.
- `curl`, `jq`, and `afplay` for the Codex speech helper.

## What Gets Installed

- App state: `~/.codex-tts`
- Command shims: `~/.local/bin/codex-tts`, `codex-speak`,
  `codex-voice-summary`
- LaunchAgent: `com.keegoid.codex-tts`
- Optional Codex config block in `~/.codex/AGENTS.md`, only after approval

Before changing an existing file, the installer writes a timestamped backup to
`~/.codex-tts/backups/<id>/`.

## Commands

```bash
codex-tts status
codex-tts start
codex-tts stop
codex-tts restart
codex-tts logs
codex-tts restore --list
codex-tts restore
codex-tts restore --backup <id>
codex-tts uninstall
```

`codex-speak "message"` is intentionally safe: if the server is offline, it
logs and exits successfully so the calling task can continue.

## API

Health:

```bash
curl -fsS http://127.0.0.1:8880/v1/health
```

Speech:

```bash
curl -fsS http://127.0.0.1:8880/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-tts","input":"Codex finished the task.","voice":"peng_mythic","response_format":"wav"}' \
  -o speech.wav
```

Public voices:

- `anime_genki`
- `anime_villain`
- `cyberpunk_cool`
- `peng_mythic`
- `anime_sultry`
- `anime_energetic`
- `anime_whisper`
- `warm_wisdom`
- `sultry_commanding`

## Development

```bash
uv sync --group dev
uv run pytest -q
uv run uvicorn codex_tts.server:app --host 127.0.0.1 --port 8880
```

For real local speech generation, install the MLX extra:

```bash
uv sync --extra mlx --group dev
```

## License

Apache-2.0. The Qwen3-TTS model used by this project is Apache-2.0 licensed.

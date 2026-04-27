# codex-tts Public Behavior Spec

## Scope

`codex-tts` is a public macOS Apple Silicon text-to-speech system for
Codex progress cues. Version 1 targets local Apple Silicon Macs and uses
the Qwen3-TTS voice-design model through MLX.

The repository must be safe to publish publicly. It must not include
private gateway clients, tokens, generated audio, local cache data, local
virtual environments, logs, personal absolute paths, or private environment
file paths.

## Server

The server exposes an OpenAI-compatible subset on `127.0.0.1:8880` by
default.

- `GET /v1/health`
  - returns JSON with `status: "ok"`, the configured TTS model id, and the
    available public voice names.
  - must not load the model just to report health.
- `POST /v1/audio/speech`
  - accepts JSON fields:
    - `model`, default `qwen3-tts`
    - `input`, required non-empty text
    - `voice`, default `peng_mythic`
    - `response_format`, default `wav`
    - `language`, default `English`
    - `instruct`, optional custom voice-design prompt
  - rejects unknown `voice` values with HTTP 400 unless `instruct` is
    provided.
  - rejects unsupported `response_format` values with HTTP 400.
  - returns generated audio bytes with the correct audio media type.
  - returns HTTP 500 if the model generates no audio or if generation fails.

Public voice names are:

- `anime_genki`
- `anime_villain`
- `cyberpunk_cool`
- `peng_mythic`
- `anime_sultry`
- `anime_energetic`
- `anime_whisper`
- `warm_wisdom`
- `sultry_commanding`

## Codex Helpers

`codex-voice-summary` is the strict helper. It:

- reads text from argv or stdin.
- trims surrounding whitespace.
- exits with code 2 when no text is provided.
- checks `curl`, `jq`, and `afplay` when playback is enabled.
- calls `/v1/health`, validates the requested voice, and sends a JSON request
  to `/v1/audio/speech`.
- supports `--voice`, `--server`, `--output`, `--no-play`, and `--help`.
- writes the output path only when `--output` is used.

`codex-speak` is the safe helper. It:

- exits successfully without speaking when called with no arguments.
- exits successfully if its strict helper is missing or not executable.
- runs the strict helper asynchronously by default.
- logs best-effort worker activity.
- never fails the caller because TTS is offline or unavailable.

## Installer

The installer is non-destructive.

- creates app state under `~/.codex-tts`.
- installs command shims in `~/.local/bin`.
- creates a launchd user service with label `com.keegoid.codex-tts`.
- asks before editing Codex config.
- if the user agrees, backs up `~/.codex/AGENTS.md` and inserts a Codex voice
  protocol block.
- before modifying any existing file, writes timestamped backups under
  `~/.codex-tts/backups/<timestamp>/`.
- supports dry-run execution that reports intended changes without modifying
  user files.

The installed `codex-tts` command supports:

- `codex-tts status`
- `codex-tts start`
- `codex-tts stop`
- `codex-tts restart`
- `codex-tts logs`
- `codex-tts restore`
- `codex-tts restore --list`
- `codex-tts restore --backup <id>`
- `codex-tts uninstall`
- `codex-tts uninstall --destroy-caches`

Restore behavior:

- `restore --list` prints available backup ids.
- `restore --backup <id>` restores from a selected backup.
- `restore` restores from the newest backup.
- restore never deletes backup sets.

Uninstall behavior:

- unloads the launchd service if present.
- removes installed command shims.
- removes launchd plist and app-managed installed files.
- leaves backups and model caches unless `--destroy-caches` is passed.

## Tests And Release Gates

Tests must cover:

- shell installer backup/restore behavior.
- shell no-argument safe helper behavior.
- mocked server behavior for `codex-voice-summary`.
- FastAPI health, voice validation, and speech error handling.
- installer dry-run behavior proving no user files are modified without backup.
- end-to-end local install with an existing local `server.py` disabled.
- sensitivity checks that fail if tracked files contain private gateway names,
  private env paths, personal absolute paths, private keys, tokens, or
  generated audio files.

Before a public push:

- run the test suite.
- run a secret scan.
- verify `git ls-files` contains no private or generated artifacts.
- verify README install commands point at `keegoid/codex-tts`.

# agent-voice Public Behavior Spec

## Scope

`agent-voice` is a public macOS Apple Silicon speech system for local agent
voice workflows. Version 1 targets local Apple Silicon Macs and
uses Qwen3-TTS voice design plus Whisper MLX speech-to-text through MLX.

The repository must be safe to publish publicly. It must not include
private gateway clients, tokens, generated audio, local cache data, local
virtual environments, logs, personal absolute paths, or private environment
file paths.

## Server

The server exposes an OpenAI-compatible subset on `127.0.0.1:8880` by
default.

- `GET /v1/health`
  - returns JSON with `status: "ok"`, the configured TTS and STT model ids, and
    the configured STT processor id and available public voice names.
  - must not load either model just to report health.
- `POST /v1/audio/speech`
  - accepts JSON fields:
    - `model`, default `qwen3-tts`
    - `input`, required non-empty text
    - `voice`, default `cyberpunk_cool`
    - `response_format`, default `wav`
    - `language`, default `English`
    - `instruct`, optional custom voice-design prompt
  - rejects unknown `voice` values with HTTP 400 unless `instruct` is
    provided.
  - rejects unsupported `response_format` values with HTTP 400.
  - returns generated audio bytes with the correct audio media type.
  - returns HTTP 500 if the model generates no audio or if generation fails.
- `POST /v1/audio/transcriptions`
  - accepts multipart form data with:
    - `file`, required audio upload.
    - `model`, default `mlx-community/whisper-large-v3-mlx`. v1 supports this
      fully qualified model id only; short aliases are rejected.
    - optional `language`, `verbose`, `max_tokens`, `chunk_duration`,
      `frame_threshold`, `context`, `prefill_step_size`, and `text`.
      These are best-effort pass-through options to the installed MLX runtime;
      unsupported options are ignored after a server-side log line.
  - rejects unsupported models with HTTP 400 before loading STT.
  - rejects uploads larger than `AGENT_VOICE_MAX_STT_UPLOAD_BYTES` before
    loading STT. The default cap is 25 MiB.
  - stores uploads in temporary files using a conservative audio suffix
    allowlist before handing them to the local MLX runtime. The service is
    loopback-only by default; it does not treat uploaded audio as trusted.
  - returns newline-delimited JSON transcription chunks as a buffered response.
    v1 does not stream partial transcription results.
  - returns HTTP 500 if STT setup or generation fails before response headers
    are sent.
  - loads `openai/whisper-large-v3` processor metadata when the MLX Whisper
    repository does not include a loadable processor.

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

## Agent Helpers

`agent-voice-summary` is the strict helper. It:

- reads text from argv or stdin.
- trims surrounding whitespace.
- exits with code 2 when no text is provided.
- checks `curl`, `jq`, and `afplay` when playback is enabled.
- calls `/v1/health`, validates the requested voice, and sends a JSON request
  to `/v1/audio/speech`.
- supports `--voice`, `--server`, `--output`, `--no-play`, and `--help`.
- writes the output path only when `--output` is used.

`agent-speak` is the safe helper. It:

- exits successfully without speaking when called with no arguments.
- exits successfully if its strict helper is missing or not executable.
- runs the strict helper asynchronously by default.
- logs best-effort worker activity.
- never fails the caller because TTS is offline or unavailable.

`codex-tts`, `codex-speak`, and `codex-voice-summary` remain compatibility
aliases for Codex-specific workflows.

## Installer

The installer is non-destructive.

- creates app state under `~/.agent-voice` for fresh installs. Existing
  `~/.codex-tts` installs keep using that directory to avoid duplicate model
  caches.
- installs command shims in `~/.local/bin`.
- creates a launchd user service with label `com.keegoid.agent-voice`.
- asks before editing Codex config.
- if the user agrees, backs up `~/.codex/AGENTS.md` and inserts a Codex voice
  protocol block.
- before modifying any existing file, writes timestamped backups under
  `~/.agent-voice/backups/<timestamp>/` or the selected legacy state directory.
- supports dry-run execution that reports intended changes without modifying
  user files.

The installed `agent-voice` command supports:

- `agent-voice status`
- `agent-voice start`
- `agent-voice stop`
- `agent-voice restart`
- `agent-voice logs`
- `agent-voice restore`
- `agent-voice restore --list`
- `agent-voice restore --backup <id>`
- `agent-voice uninstall`
- `agent-voice uninstall --destroy-caches`

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
- mocked server behavior for `agent-voice-summary`.
- FastAPI health, voice validation, speech error handling, and transcription
  contract behavior.
- installer dry-run behavior proving no user files are modified without backup.
- end-to-end local install with an existing local `server.py` disabled.
- sensitivity checks that fail if tracked files contain private gateway names,
  private env paths, personal absolute paths, private keys, tokens, or
  generated audio files.

Before a public push:

- run the test suite.
- run a secret scan.
- verify `git ls-files` contains no private or generated artifacts.
- verify README install commands point at `keegoid/agent-voice`.

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
    the configured STT processor id, current `muted` state, and available
    public voice names.
  - must not load either model just to report health.
- `GET /v1/mute`
  - returns the persistent master mute state.
- `POST /v1/mute`
  - accepts JSON `{ "muted": true|false }` and persists the master mute state
    under `AGENT_VOICE_HOME` unless `AGENT_VOICE_MUTE_STATE` overrides the
    path.
- `POST /v1/mute/toggle`
  - flips and persists the master mute state.
- `POST /v1/audio/speech`
  - accepts JSON fields:
    - `model`, default `qwen3-tts`
    - `input`, required non-empty text
    - `voice`, default `cyberpunk_cool`
    - `response_format`, default `wav`
    - `language`, default `English`
    - `instruct`, optional custom voice-design prompt
    - `max_tokens`, optional TTS generation budget
  - rejects unknown `voice` values with HTTP 400 unless `instruct` is
    provided.
  - supports `response_format` values `wav`, `mp3`, `opus`, and `flac`.
    MP3 and Opus output require `ffmpeg`; WAV and FLAC do not.
  - rejects other `response_format` values with HTTP 400.
  - when master mute is enabled, returns valid silent audio for the requested
    response format without loading or calling the TTS model.
  - does not impose a request character cap. Long requests may be split into
    multiple synthesis segments server-side and concatenated into one response
    audio file.
  - defaults to `AGENT_VOICE_TTS_MAX_TOKENS=24000` and retries suspiciously
    short generated segments once by default, including terse status cues and
    clips that speak only at the beginning then continue as silence.
  - uses contiguous active-speech duration, not total audio file length, for
    suspiciously short segment detection.
  - inserts `AGENT_VOICE_TTS_SEGMENT_SILENCE_SECONDS=0.18` seconds of silence
    between generated segments by default.
  - returns generated audio bytes with the correct audio media type.
  - returns HTTP 500 if the model generates no audio or if generation fails.
- `POST /v1/audio/transcriptions`
  - accepts multipart form data with:
    - `file`, required audio upload.
    - `model`, default `mlx-community/whisper-large-v3-mlx`. v1 supports this
      fully qualified model id only; short aliases are rejected.
    - optional `language`, `verbose`, `max_tokens`, `chunk_duration`,
      `context`, and `text`.
      These are best-effort pass-through options to the installed MLX runtime;
      unsupported options are ignored after a server-side log line.
    - legacy `frame_threshold` and `prefill_step_size` fields are accepted for
      old PAI/Fig callers but hidden from the schema and not forwarded to MLX
      Whisper v1 because the current runtime rejects them.
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
  - treats `AGENT_VOICE_STT_PROCESSOR_ID` as trust-sensitive: it controls which
    Hugging Face processor repository is loaded for Whisper compatibility.

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
- exits successfully without generating or playing audio when `/v1/health`
  reports `muted: true`.
- supports `--voice`, `--server`, `--output`, `--play-timeout`, `--no-play`,
  and `--help`.
- bounds `afplay` runtime with a duration-aware timeout by default. The
  `AGENT_VOICE_PLAYBACK_TIMEOUT_SECONDS` environment variable overrides that
  default.
- writes the output path only when `--output` is used.

`agent-speak` is the safe helper. It:

- exits successfully without speaking when called with no arguments.
- exits successfully if its strict helper is missing or not executable.
- runs the strict helper asynchronously by default.
- logs best-effort worker activity.
- never fails the caller because TTS is offline or unavailable.

Configuration uses `AGENT_VOICE_*` environment variables.

## Installer

The installer is non-destructive.

- creates app state under `~/.agent-voice`.
- installs command shims in `~/.local/bin`.
- creates a launchd user service with label `com.keegoid.agent-voice`.
- asks before editing Codex config.
- if the user agrees, backs up `~/.codex/AGENTS.md` and inserts a Codex voice
  protocol block.
- before modifying any existing file, writes timestamped backups under
  `~/.agent-voice/backups/<timestamp>/`.
- supports dry-run execution that reports intended changes without modifying
  user files.

The installed `agent-voice` command supports:

- `agent-voice status`
- `agent-voice start`
- `agent-voice stop`
- `agent-voice restart`
- `agent-voice logs`
- `agent-voice mute`
- `agent-voice mute status`
- `agent-voice mute toggle`
- `agent-voice unmute`
- `agent-voice configure hermes`
- `agent-voice restore`
- `agent-voice restore --list`
- `agent-voice restore --backup <id>`
- `agent-voice uninstall`
- `agent-voice uninstall --destroy-caches`

Hermes configuration behavior:

- `agent-voice configure hermes` updates `~/.hermes/config.yaml` so Hermes uses
  the local OpenAI-compatible speech endpoint:
  - `tts.provider: openai`
  - `tts.openai.model: qwen3-tts` unless overridden.
  - `tts.openai.voice: cyberpunk_cool` unless overridden.
  - `tts.openai.base_url: http://127.0.0.1:8880/v1` unless overridden.
  - `voice.auto_tts: true` by default.
- It writes `VOICE_TOOLS_OPENAI_KEY=agent-voice-local` to `~/.hermes/.env`
  when that key is missing or empty. Existing non-empty values are preserved.
- It backs up existing Hermes config files before editing.
- `--dry-run` prints intended changes without modifying files.
- `--restart-gateway` restarts the Hermes gateway after successful writes.
- `--hermes-home` must point inside `$HOME` unless `--allow-outside-home` is
  set.

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

# agent-voice

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-black.svg)](#requirements)
[![Python](https://img.shields.io/badge/python-3.12%2B-3776AB.svg)](pyproject.toml)
[![API](https://img.shields.io/badge/API-OpenAI%20speech%20subset-009688.svg)](#api)

Local speech for agent workflows on macOS Apple Silicon. It serves small
OpenAI-compatible speech and transcription endpoints backed by Qwen3-TTS and
Whisper through MLX, plus shell helpers that make voice cues best-effort instead
of task-breaking.

Voice cues make agentic workflows easier to follow when your eyes are elsewhere.
You can be on another screen, or listening from the kitchen, and still know when
an agent starts something risky, gets blocked, or finishes a useful result. The
voice-design prompts also add personality: because generation is probabilistic,
the same preset can sometimes land with surprising energy, emotion, or timing.

## Install

Version 1 supports macOS Apple Silicon only.

```bash
tmp="$(mktemp -d)"
git clone --depth 1 --branch v0.2.1 https://github.com/keegoid/agent-voice "$tmp/agent-voice"
"$tmp/agent-voice/install.sh"
```

The convenience command does not use shell piping, and it targets a release tag
instead of the moving default branch. For a stricter install, clone or download a
pinned commit, inspect it, then run `./install.sh`. Remote archive installs
require `--archive-sha256 <sha256>`.

## Requirements

- macOS on Apple Silicon.
- Python 3.12 or newer.
- `uv`.
- `git` for the convenience install command.
- `curl`, `jq`, and `afplay` for the shell speech helpers.
- `ffmpeg` for OpenAI-compatible MP3 and Opus speech responses. WAV and FLAC
  responses do not need transcoding.
- Network access for the first install. The installer creates a local virtual
  environment and downloads the MLX runtime/model dependencies into app-managed
  state.
- Disk and memory headroom. Hugging Face currently lists the Qwen3-TTS
  VoiceDesign bf16 repository at about 4.5 GB, and the optional Whisper
  large-v3 MLX STT model at about 3.1 GB. First use downloads model files into
  `~/.agent-voice/model-cache`; runtime inference also needs several GB of unified
  memory, so low-memory Macs may swap or fail under load.

## What Gets Installed

- App state: `~/.agent-voice`
- Command shims: `~/.local/bin/agent-voice`, `agent-speak`,
  `agent-voice-summary`
- LaunchAgent: `com.keegoid.agent-voice`
- Optional Codex config block in `~/.codex/AGENTS.md`, only after approval

Before changing an existing file, the installer writes a timestamped backup to
`~/.agent-voice/backups/<id>/`.

## Model Download

The installer does not download the Qwen3-TTS or Whisper model directly. It
installs the Python runtime and MLX dependencies, then launchd starts the local
server. The server downloads and loads
`mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16` the first time
`/v1/audio/speech` needs generation, and
`mlx-community/whisper-large-v3-mlx` the first time `/v1/audio/transcriptions`
needs transcription. `/v1/health` stays cheap and does not load either model.
The Whisper MLX repository does not ship the processor metadata that
`mlx-audio` expects, so the server also loads the small
`openai/whisper-large-v3` processor on first transcription.
`AGENT_VOICE_STT_PROCESSOR_ID` can override that processor repository for local
experiments; treat it as trust-sensitive because it controls Hugging Face model
loading.

Speech generation defaults to `AGENT_VOICE_TTS_MAX_TOKENS=24000`. There is no
request character cap; long requests are split into bounded synthesis segments
and concatenated so agent summaries can stay useful without caller-side
trimming. If a generated segment is implausibly short for the text length, the
server retries it once with more conservative sampling. The short-clip detector
also covers terse agent status cues and clips that speak only at the beginning
then continue as silence. Tune it with
`AGENT_VOICE_TTS_SUSPICIOUS_MIN_WORDS`,
`AGENT_VOICE_TTS_SUSPICIOUS_MAX_WORDS_PER_SECOND`, and
`AGENT_VOICE_TTS_SUSPICIOUS_MIN_SECONDS` if needed. The detector measures
the longest contiguous active-speech span using `AGENT_VOICE_TTS_ACTIVITY_WINDOW_SECONDS`,
`AGENT_VOICE_TTS_ACTIVITY_MIN_RMS`, and
`AGENT_VOICE_TTS_ACTIVITY_RELATIVE_RMS`.
Segment joins use `AGENT_VOICE_TTS_SEGMENT_SILENCE_SECONDS=0.18` by default.

The launchd service sets `HF_HOME` to
`~/.agent-voice/model-cache/huggingface`, so Hugging Face/MLX model files are
kept under `~/.agent-voice/model-cache` for fresh installs.

## Commands

```bash
agent-voice status
agent-voice start
agent-voice stop
agent-voice restart
agent-voice logs
agent-voice mute
agent-voice mute status
agent-voice mute toggle
agent-voice unmute
agent-voice configure hermes
agent-voice restore --list
agent-voice restore
agent-voice restore --backup <id>
agent-voice uninstall
```

`agent-speak "message"` is intentionally safe: if the server is offline, it
logs and exits successfully so the calling task can continue.
`agent-voice mute` is a persistent master mute for local speech generation.
While muted, `/v1/audio/speech` returns valid silent audio without loading the
TTS model, and `agent-voice-summary` exits before generating or playing audio.
Use `AGENT_VOICE_*` environment variables for configuration.
Playback is bounded separately from generation: `agent-voice-summary` defaults
to a playback timeout based on the generated WAV duration plus grace, and
`AGENT_VOICE_PLAYBACK_TIMEOUT_SECONDS` or `--play-timeout` can override it.

Uninstall removes only shims that point at the managed `agent-voice` install. If
an earlier shim was backed up during install, uninstall restores that previous
file instead of deleting it.

## API

Health:

```bash
curl -fsS http://127.0.0.1:8880/v1/health
```

PAI-compatible desktop notification and playback:

```bash
curl -fsS http://127.0.0.1:8880/notify \
  -H 'Content-Type: application/json' \
  -d '{"title":"PAI Notification","message":"The agent finished the task.","voice_id":"cyberpunk_cool","voice_enabled":true}'
```

Compatibility aliases are also available at `/notify/personality` and `/pai`.
These endpoints display a macOS desktop notification, then generate WAV speech
with the same Qwen3 runtime and play it locally with `afplay`. They preserve old
PAI notification payload fields: `title`, `message`, `voice_id`, `voice_name`,
`voice_enabled`, optional `instruct`, and `language`. Unknown old voice ids fall
back to the configured default voice unless a custom `instruct` is supplied.
Pronunciation replacements are loaded from
`~/.agent-voice/pronunciations.json`, or from `AGENT_VOICE_PRONUNCIATIONS_PATH`
when set:

```json
{
  "replacements": [
    { "term": "PAI", "phonetic": "pie" },
    { "term": "ISC", "phonetic": "I S C" }
  ]
}
```

Notify playback is serialized and bounded by
`AGENT_VOICE_NOTIFY_QUEUE_MAX_DEPTH=3` by default. Tune defaults with
`AGENT_VOICE_NOTIFY_DEFAULT_VOICE`, `AGENT_VOICE_NOTIFY_DESKTOP`,
`AGENT_VOICE_NOTIFY_RATE_LIMIT`, and `AGENT_VOICE_NOTIFY_PLAYBACK_TIMEOUT_SECONDS`.

Mute state:

```bash
curl -fsS http://127.0.0.1:8880/v1/mute
curl -fsS http://127.0.0.1:8880/v1/mute \
  -H 'Content-Type: application/json' \
  -d '{"muted":true}'
curl -fsS -X POST http://127.0.0.1:8880/v1/mute/toggle
```

Speech:

```bash
curl -fsS http://127.0.0.1:8880/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-tts","input":"The agent finished the task.","voice":"cyberpunk_cool","response_format":"wav","max_tokens":24000}' \
  -o speech.wav
```

OpenAI-compatible speech `response_format` values:

- `wav`
- `mp3`
- `opus`
- `flac`

MP3 and Opus responses require `ffmpeg` on `PATH`.

The server uses stable Qwen3 sampling defaults for speech:
`AGENT_VOICE_TTS_TEMPERATURE` (default `0.9`),
`AGENT_VOICE_TTS_TOP_P` (default `0.95`), and
`AGENT_VOICE_TTS_REPETITION_PENALTY` (default `1.05`). Retry attempts use the
`AGENT_VOICE_TTS_RETRY_*` variants. Before encoding, generated waveforms are
sanitized and peak-limited with `AGENT_VOICE_TTS_PEAK_LIMIT` (default `0.98`).

Transcription:

```bash
curl -fsS http://127.0.0.1:8880/v1/audio/transcriptions \
  -F file=@speech.wav \
  -F model=mlx-community/whisper-large-v3-mlx \
  -F language=en
```

The transcription response is newline-delimited JSON, buffered before it is sent.
Version 1 does not stream partial transcription results. The `model` field must
be `mlx-community/whisper-large-v3-mlx`; short aliases are rejected so callers do
not accidentally invoke a different local model. Advanced generation knobs are
passed through to the installed MLX runtime when supported and ignored with a
server-side log line when that runtime does not accept them.
Legacy PAI/Fig fields `frame_threshold` and `prefill_step_size` are accepted for
compatibility but are not forwarded to the current MLX Whisper runtime.

Public voices:

- `anime_genki`
- `anime_villain`
- `cool_figment_rain_voice_locked`
- `cool_street_deadpan_voice_locked`
- `cyberpunk_cool`
- `peng_mythic`
- `anime_sultry`
- `anime_energetic`
- `anime_whisper`
- `warm_wisdom`
- `sultry_commanding`

## Hermes Agent

Hermes Agent can use `agent-voice` through its existing OpenAI-compatible TTS
backend. No Hermes code patch is required; point Hermes at the local server and
give it any non-empty audio API key.

With `agent-voice` installed and running, configure the default Hermes home:

```bash
agent-voice configure hermes --restart-gateway
```

Preview first:

```bash
agent-voice configure hermes --dry-run
```

That command backs up the existing Hermes files, then writes this TTS config to
`~/.hermes/config.yaml`:

```yaml
tts:
  provider: openai
  openai:
    model: qwen3-tts
    voice: cyberpunk_cool
    base_url: http://127.0.0.1:8880/v1

voice:
  auto_tts: true
```

It also ensures `~/.hermes/.env` contains:

```bash
VOICE_TOOLS_OPENAI_KEY=agent-voice-local
```

The key is only a local bearer token placeholder; `agent-voice` does not require
a real OpenAI key when bound to loopback. Hermes requests MP3 for normal output
and Opus for Telegram-style voice messages, both of which `agent-voice`
supports. MP3 and Opus require `ffmpeg` on `PATH`.

Manual setup is the same edit: set the `tts` block above, set
`VOICE_TOOLS_OPENAI_KEY`, then restart any running Hermes gateway:

```bash
hermes gateway restart
```

See [docs/hermes.md](docs/hermes.md) for restore options and extra flags such as
`--hermes-home`, `--voice`, and `--auto-tts false`.

## Development

```bash
uv sync --group dev
uv run pytest -q
uv run uvicorn agent_voice.server:app --host 127.0.0.1 --port 8880
```

Agent-authored branch, PR, and review work should go through the shared
Keegoid workflow helper instead of hand-rolled git/GitHub commands:

```bash
agent-pr-flow begin --actor codex --branch <slug>
agent-pr-flow commit --actor codex --all --message "<message>"
agent-pr-flow publish --actor codex --title "<title>" --body-file <file>
```

After a PR is merged and local `main` is synced, update the installed launchd
runtime from the checked-out source with:

```bash
scripts/sync-installed --from main --test
```

The sync command refuses dirty source trees by default, runs the installer
non-interactively, writes `~/.agent-voice/install-manifest.json`, restarts via
the installer, and verifies `/v1/health`.

For real local speech generation, install the MLX extra:

```bash
uv sync --extra mlx --group dev
```

## License

Apache-2.0. The Qwen3-TTS model used by this project is Apache-2.0 licensed.

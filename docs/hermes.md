# Hermes Agent Integration

`agent-voice` exposes the OpenAI-compatible speech endpoint shape that Hermes
Agent already knows how to call. The integration does not require a real OpenAI
audio key because the server is local and unauthenticated on loopback.

## One-command Setup

With `agent-voice` installed and running:

```bash
agent-voice configure hermes --restart-gateway
```

The command backs up existing Hermes files under
`~/.agent-voice/backups/<timestamp>-hermes-<pid>/`, then updates:

- `~/.hermes/config.yaml`
  - `tts.provider: openai`
  - `tts.openai.model: qwen3-tts`
  - `tts.openai.voice: cyberpunk_cool`
  - `tts.openai.base_url: http://127.0.0.1:8880/v1`
  - `voice.auto_tts: true`
- `~/.hermes/.env`
  - `VOICE_TOOLS_OPENAI_KEY=agent-voice-local`

Preview without writing:

```bash
agent-voice configure hermes --dry-run
```

Use a different preset voice:

```bash
agent-voice configure hermes --voice warm_wisdom --restart-gateway
```

## Manual Configuration

Edit `~/.hermes/config.yaml`:

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

Edit `~/.hermes/.env`:

```bash
VOICE_TOOLS_OPENAI_KEY=agent-voice-local
```

Then restart the gateway:

```bash
hermes gateway restart
```

## Notes

- Hermes usually requests MP3 for Discord and desktop output. `agent-voice`
  supports `response_format: mp3` directly.
- Telegram-style Opus output is supported with `response_format: opus` or
  `response_format: ogg`.
- MP3 and Opus output require `ffmpeg` on `PATH`. WAV and FLAC do not.
- `voice.auto_tts: true` makes Hermes speak replies to voice-message flows. For
  voice attachments on every text reply in a specific messaging channel, run
  `/voice tts` in that channel.

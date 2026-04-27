"""Compatibility module for older ``codex_tts.server`` launchers."""

from agent_voice.server import app, audio_speech, audio_transcriptions, health, main

__all__ = ["app", "audio_speech", "audio_transcriptions", "health", "main"]


if __name__ == "__main__":
    main()

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from tests.helpers.public_contract import REPO_ROOT


TEXT_EXTENSIONS = {
    ".bash",
    ".cfg",
    ".css",
    "." + "env",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".ts",
    ".txt",
    ".yaml",
    ".yml",
}

AUDIO_EXTENSIONS = {".aif", ".aiff", ".flac", ".m4a", ".mp3", ".ogg", ".wav"}


def tracked_files() -> list[Path]:
    result = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, text=True, capture_output=True, check=True)
    return [path for line in result.stdout.splitlines() if line and (path := REPO_ROOT / line).exists()]


def text_files() -> list[Path]:
    return [path for path in tracked_files() if path.suffix.lower() in TEXT_EXTENSIONS]


def test_tracked_files_do_not_include_generated_audio_artifacts() -> None:
    audio_files = [path.relative_to(REPO_ROOT) for path in tracked_files() if path.suffix.lower() in AUDIO_EXTENSIONS]

    assert audio_files == []


def test_tracked_text_does_not_reference_retired_private_gateway_or_local_paths() -> None:
    forbidden_literals = [
        "OPEN" + "CLAW",
        "/" + "Users" + "/" + "kmullaney",
        "codex" + "-tts",
        "codex" + "_tts",
        "CODEX" + "_TTS",
        "." + "codex" + "-tts",
        "codex" + "-speak",
        "codex" + "-voice",
        "CODEX" + "_SPEAK",
        "CODEX" + "_VOICE",
    ]
    violations: list[str] = []
    for path in text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for literal in forbidden_literals:
            if literal in text:
                violations.append(f"{path.relative_to(REPO_ROOT)} contains {literal}")

    assert violations == []


def test_tracked_files_do_not_include_environment_files() -> None:
    forbidden_name = "." + "env"
    env_files = [
        path.relative_to(REPO_ROOT)
        for path in tracked_files()
        if path.name == forbidden_name or path.name.endswith(forbidden_name)
    ]

    assert env_files == []


def test_tracked_text_does_not_contain_private_key_material() -> None:
    private_key_pattern = re.compile(r"-{5}BEGIN [A-Z ]*PRIVATE KEY-{5}")
    violations = [
        str(path.relative_to(REPO_ROOT))
        for path in text_files()
        if private_key_pattern.search(path.read_text(encoding="utf-8", errors="ignore"))
    ]

    assert violations == []


def test_tracked_text_does_not_contain_token_assignments() -> None:
    token_pattern = re.compile(
        r"(?i)(api|access|auth|bearer|secret|session)[_-]?(key|token)\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{16,}"
    )
    violations = [
        str(path.relative_to(REPO_ROOT))
        for path in text_files()
        if token_pattern.search(path.read_text(encoding="utf-8", errors="ignore"))
    ]

    assert violations == []

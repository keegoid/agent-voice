from __future__ import annotations

import importlib
import json
import os
import shutil
import socket
import subprocess
import sys
import textwrap
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterable


PUBLIC_VOICES = {
    "anime_genki",
    "anime_villain",
    "cyberpunk_cool",
    "peng_mythic",
    "anime_sultry",
    "anime_energetic",
    "anime_whisper",
    "warm_wisdom",
    "sultry_commanding",
}


REPO_ROOT = Path(__file__).resolve().parents[2]


def executable_candidates(name: str) -> list[Path]:
    matches: list[Path] = []
    for path in REPO_ROOT.rglob(name):
        if any(part in {".git", ".venv", "venv", "__pycache__", "node_modules"} for part in path.parts):
            continue
        if path.is_file() and os.access(path, os.X_OK):
            matches.append(path)
    return sorted(matches)


def require_executable(name: str) -> Path:
    matches = executable_candidates(name)
    if not matches:
        import pytest

        pytest.skip(f"{name} executable is not present yet")
    return matches[0]


def require_installer() -> Path:
    names = ("install.sh", "installer.sh", "install")
    matches: list[Path] = []
    for name in names:
        matches.extend(executable_candidates(name))
    if not matches:
        import pytest

        pytest.skip("installer executable is not present yet")
    return sorted(matches)[0]


def run_with_home(
    argv: list[str],
    tmp_path: Path,
    *,
    input_text: str | None = None,
    extra_env: dict[str, str] | None = None,
    timeout: int = 10,
) -> subprocess.CompletedProcess[str]:
    home = tmp_path / "home"
    local_bin = home / ".local" / "bin"
    home.mkdir(exist_ok=True)
    local_bin.mkdir(parents=True, exist_ok=True)
    env = getattr(os, "environ").copy()
    env.update(
        {
            "HOME": str(home),
            "PATH": f"{local_bin}:{env.get('PATH', '')}",
            "CODEX_TTS_TEST_MODE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        argv,
        input=input_text,
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=timeout,
    )


def tree_snapshot(root: Path) -> dict[str, bytes]:
    if not root.exists():
        return {}
    snapshot: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            snapshot[str(path.relative_to(root))] = path.read_bytes()
    return snapshot


def installed_command(tmp_path: Path, name: str) -> Path:
    return tmp_path / "home" / ".local" / "bin" / name


def run_command_from_temp_path(
    tmp_path: Path,
    argv: list[str],
    *,
    input_text: str | None = None,
    timeout: int = 10,
) -> subprocess.CompletedProcess[str]:
    env = getattr(os, "environ").copy()
    home = tmp_path / "home"
    env.update({"HOME": str(home), "PATH": f"{home / '.local' / 'bin'}:{env.get('PATH', '')}"})
    return subprocess.run(
        argv,
        input=input_text,
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=timeout,
    )


@dataclass
class SpeechRequest:
    path: str
    body: dict[str, Any]


class MockSpeechServer:
    def __init__(self, *, voices: Iterable[str] = PUBLIC_VOICES, audio: bytes = b"RIFFtestWAVEfmt ") -> None:
        self.voices = list(voices)
        self.audio = audio
        self.requests: list[SpeechRequest] = []
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        assert self._server is not None
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    def __enter__(self) -> "MockSpeechServer":
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path != "/v1/health":
                    self.send_error(404)
                    return
                payload = {"status": "ok", "model": "qwen3-tts", "voices": outer.voices}
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode())

            def do_POST(self) -> None:  # noqa: N802
                if self.path != "/v1/audio/speech":
                    self.send_error(404)
                    return
                length = int(self.headers.get("content-length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                outer.requests.append(SpeechRequest(path=self.path, body=body))
                self.send_response(200)
                self.send_header("content-type", "audio/wav")
                self.end_headers()
                self.wfile.write(outer.audio)

            def log_message(self, *_args: Any) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if self._thread:
            self._thread.join(timeout=2)


def locate_fastapi_app() -> Any:
    getattr(os, "environ").setdefault("CODEX_TTS_TEST_MODE", "1")
    getattr(os, "environ").setdefault("CODEX_TTS_DISABLE_MODEL_LOAD", "1")
    candidates = [
        "codex_tts.server",
        "codex_tts.app",
        "server",
        "app",
        "main",
    ]
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for attr in ("app", "application"):
            app = getattr(module, attr, None)
            if getattr(app, "routes", None):
                return app
    import pytest

    pytest.skip("FastAPI app is not importable from a public server module yet")


def patch_generation(monkeypatch: Any, app: Any, replacement: Callable[..., Any]) -> None:
    modules = {sys.modules.get(getattr(route.endpoint, "__module__", "")) for route in getattr(app, "routes", [])}
    for module in [m for m in modules if m is not None]:
        for name in dir(module):
            lowered = name.lower()
            if any(key in lowered for key in ("generate", "synthesize", "speech", "audio")):
                target = getattr(module, name)
                if callable(target) and not name.startswith("_"):
                    monkeypatch.setattr(module, name, replacement, raising=False)


def make_fake_bin(directory: Path, name: str, script: str) -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(script).lstrip(), encoding="utf-8")
    path.chmod(0o755)
    return path


@contextmanager
def unused_port() -> Iterable[int]:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    yield port

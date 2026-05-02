"""Persistent master mute state for local agent-voice speech."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOCK = threading.RLock()
_TRUE_VALUES = {"1", "true", "yes", "on", "muted"}
_FALSE_VALUES = {"0", "false", "no", "off", "unmuted"}


class MuteStateLockedError(RuntimeError):
    """Raised when an environment override owns mute state."""


def state_dir() -> Path:
    return Path(os.environ.get("AGENT_VOICE_HOME") or Path.home() / ".agent-voice").expanduser()


def state_path() -> Path:
    override = os.environ.get("AGENT_VOICE_MUTE_STATE")
    if override:
        return Path(override).expanduser()
    return state_dir() / "mute.json"


def _env_override() -> bool | None:
    raw = os.environ.get("AGENT_VOICE_MUTED")
    if raw is None:
        return None
    normalized = raw.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return None


def env_override_active() -> bool:
    return _env_override() is not None


def read_state() -> dict[str, Any]:
    override = _env_override()
    if override is not None:
        return {"muted": override, "source": "env", "path": str(state_path())}

    path = state_path()
    if os.environ.get("AGENT_VOICE_TEST_MODE") == "1" and not os.environ.get("AGENT_VOICE_MUTE_STATE"):
        return {"muted": False, "source": "test-default", "path": str(path)}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"muted": False, "source": "default", "path": str(path)}
    except (OSError, ValueError, TypeError) as exc:
        return {
            "muted": False,
            "source": "error",
            "path": str(path),
            "error": f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(data, dict):
        return {"muted": False, "source": "invalid", "path": str(path)}
    data = dict(data)
    data["muted"] = bool(data.get("muted"))
    data.setdefault("source", "file")
    data["path"] = str(path)
    return data


def is_muted() -> bool:
    return bool(read_state().get("muted"))


def set_muted(muted: bool, *, source: str = "api") -> dict[str, Any]:
    if env_override_active():
        raise MuteStateLockedError("AGENT_VOICE_MUTED overrides persisted mute state")

    path = state_path()
    payload: dict[str, Any] = {
        "muted": bool(muted),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp:
                json.dump(payload, tmp, sort_keys=True)
                tmp.write("\n")
            os.replace(tmp_name, path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
    payload["path"] = str(path)
    return payload


def toggle_muted(*, source: str = "api") -> dict[str, Any]:
    if env_override_active():
        raise MuteStateLockedError("AGENT_VOICE_MUTED overrides persisted mute state")

    with _LOCK:
        return set_muted(not is_muted(), source=source)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent_voice.mute_state")
    parser.add_argument("action", nargs="?", choices=("status", "on", "off", "toggle"), default="status")
    args = parser.parse_args(argv)

    try:
        if args.action == "on":
            state = set_muted(True, source="cli")
        elif args.action == "off":
            state = set_muted(False, source="cli")
        elif args.action == "toggle":
            state = toggle_muted(source="cli")
        else:
            state = read_state()
    except MuteStateLockedError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print("muted" if state.get("muted") else "unmuted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import os
from pathlib import Path

from tests.helpers.public_contract import (
    installed_command,
    make_fake_bin,
    require_installer,
    run_command_from_temp_path,
    run_with_home,
    tree_snapshot,
)


def test_installer_dry_run_does_not_modify_existing_user_files(tmp_path: Path) -> None:
    installer = require_installer()
    home = tmp_path / "home"
    agents = home / ".codex" / "AGENTS.md"
    shim = home / ".local" / "bin" / "codex-speak"
    agents.parent.mkdir(parents=True)
    shim.parent.mkdir(parents=True)
    agents.write_text("operator notes\n", encoding="utf-8")
    shim.write_text("#!/bin/sh\necho existing\n", encoding="utf-8")
    before = tree_snapshot(home)

    result = run_with_home([str(installer), "--dry-run"], tmp_path, input_text="y\n")

    assert result.returncode == 0, result.stderr
    assert tree_snapshot(home) == before
    assert "dry" in (result.stdout + result.stderr).lower()


def test_installer_backs_up_agents_before_restore(tmp_path: Path) -> None:
    installer = require_installer()
    home = tmp_path / "home"
    fake_bin = tmp_path / "fake-bin"
    make_fake_bin(fake_bin, "launchctl", "#!/bin/sh\nexit 0\n")
    agents = home / ".codex" / "AGENTS.md"
    agents.parent.mkdir(parents=True)
    agents.write_text("original codex config\n", encoding="utf-8")

    install = run_with_home(
        [str(installer)],
        tmp_path,
        input_text="y\n",
        extra_env={"PATH": f"{fake_bin}:{home / '.local' / 'bin'}:{getattr(os, 'environ').get('PATH', '')}"},
        timeout=20,
    )

    assert install.returncode == 0, install.stderr
    backups = sorted((home / ".codex-tts" / "backups").glob("*"))
    assert backups, "installer must create a timestamped backup set before editing"
    assert any(path.name == "AGENTS.md" and path.read_text(encoding="utf-8") == "original codex config\n" for path in backups[0].rglob("*"))

    command = installed_command(tmp_path, "codex-tts")
    if not command.exists():
        return
    restore = run_command_from_temp_path(tmp_path, [str(command), "restore"], timeout=20)
    assert restore.returncode == 0, restore.stderr
    assert agents.read_text(encoding="utf-8") == "original codex config\n"
    assert backups[0].exists(), "restore must not delete backup sets"


def test_end_to_end_install_disables_existing_local_server_py(tmp_path: Path) -> None:
    installer = require_installer()
    home = tmp_path / "home"
    fake_bin = tmp_path / "fake-bin"
    make_fake_bin(fake_bin, "launchctl", "#!/bin/sh\nexit 0\n")
    stale_server = home / ".codex-tts" / "server.py"
    stale_server.parent.mkdir(parents=True)
    stale_server.write_text("raise SystemExit('stale local server')\n", encoding="utf-8")

    result = run_with_home(
        [str(installer)],
        tmp_path,
        input_text="n\n",
        extra_env={"PATH": f"{fake_bin}:{home / '.local' / 'bin'}:{getattr(os, 'environ').get('PATH', '')}"},
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert not stale_server.exists() or "stale local server" not in stale_server.read_text(encoding="utf-8")
    backup_root = home / ".codex-tts" / "backups"
    assert any(path.name == "server.py" for path in backup_root.rglob("*"))


def test_uninstall_restores_preexisting_command_shim(tmp_path: Path) -> None:
    installer = require_installer()
    home = tmp_path / "home"
    fake_bin = tmp_path / "fake-bin"
    make_fake_bin(fake_bin, "launchctl", "#!/bin/sh\nexit 0\n")
    original = home / ".local" / "bin" / "codex-speak"
    original.parent.mkdir(parents=True)
    original.write_text("#!/bin/sh\necho original speaker\n", encoding="utf-8")
    original.chmod(0o755)

    install = run_with_home(
        [str(installer)],
        tmp_path,
        input_text="n\n",
        extra_env={"PATH": f"{fake_bin}:{home / '.local' / 'bin'}:{getattr(os, 'environ').get('PATH', '')}"},
        timeout=20,
    )

    assert install.returncode == 0, install.stderr
    assert "original speaker" not in original.read_text(encoding="utf-8")

    reinstall = run_with_home(
        [str(installer)],
        tmp_path,
        input_text="n\n",
        extra_env={"PATH": f"{fake_bin}:{home / '.local' / 'bin'}:{getattr(os, 'environ').get('PATH', '')}"},
        timeout=20,
    )

    assert reinstall.returncode == 0, reinstall.stderr

    command = installed_command(tmp_path, "codex-tts")
    uninstall = run_with_home(
        [str(command), "uninstall"],
        tmp_path,
        extra_env={"PATH": f"{fake_bin}:{home / '.local' / 'bin'}:{getattr(os, 'environ').get('PATH', '')}"},
        timeout=20,
    )

    assert uninstall.returncode == 0, uninstall.stderr
    assert original.read_text(encoding="utf-8") == "#!/bin/sh\necho original speaker\n"


def test_uninstall_does_not_restore_managed_shim_when_no_original_exists(tmp_path: Path) -> None:
    installer = require_installer()
    home = tmp_path / "home"
    fake_bin = tmp_path / "fake-bin"
    make_fake_bin(fake_bin, "launchctl", "#!/bin/sh\nexit 0\n")
    env = {"PATH": f"{fake_bin}:{home / '.local' / 'bin'}:{getattr(os, 'environ').get('PATH', '')}"}

    first = run_with_home([str(installer)], tmp_path, input_text="n\n", extra_env=env, timeout=20)
    assert first.returncode == 0, first.stderr

    second = run_with_home([str(installer)], tmp_path, input_text="n\n", extra_env=env, timeout=20)
    assert second.returncode == 0, second.stderr

    command = installed_command(tmp_path, "codex-tts")
    uninstall = run_with_home([str(command), "uninstall"], tmp_path, extra_env=env, timeout=20)

    assert uninstall.returncode == 0, uninstall.stderr
    assert not installed_command(tmp_path, "codex-speak").exists()


def test_installer_fails_when_launchd_bootstrap_fails(tmp_path: Path) -> None:
    installer = require_installer()
    home = tmp_path / "home"
    fake_bin = tmp_path / "fake-bin"
    make_fake_bin(
        fake_bin,
        "uname",
        """
        #!/bin/sh
        case "$1" in
          -s) echo Darwin ;;
          -m) echo arm64 ;;
        esac
        """,
    )
    make_fake_bin(
        fake_bin,
        "uv",
        """
        #!/bin/sh
        project=""
        while [ "$#" -gt 0 ]; do
          if [ "$1" = "--project" ]; then
            project="$2"
            shift 2
          else
            shift
          fi
        done
        mkdir -p "$project/.venv/bin"
        printf '#!/bin/sh\n' > "$project/.venv/bin/python"
        chmod +x "$project/.venv/bin/python"
        """,
    )
    make_fake_bin(
        fake_bin,
        "launchctl",
        """
        #!/bin/sh
        case "$1" in
          bootout) exit 0 ;;
          bootstrap) echo "bootstrap denied" >&2; exit 42 ;;
          print) exit 42 ;;
          kickstart) exit 42 ;;
        esac
        exit 0
        """,
    )

    result = run_with_home(
        [str(installer)],
        tmp_path,
        input_text="n\n",
        extra_env={
            "CODEX_TTS_TEST_MODE": "0",
            "PATH": f"{fake_bin}:{home / '.local' / 'bin'}:{getattr(os, 'environ').get('PATH', '')}",
        },
        timeout=20,
    )

    assert result.returncode != 0
    assert "bootstrap denied" in result.stderr

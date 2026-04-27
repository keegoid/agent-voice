#!/usr/bin/env bash
# Non-destructive installer for agent-voice.

set -euo pipefail

REPO_URL="https://github.com/keegoid/agent-voice"
ARCHIVE_REF="${AGENT_VOICE_REF:-main}"
ARCHIVE_URL="$REPO_URL/archive/$ARCHIVE_REF.tar.gz"
ARCHIVE_SHA256="${AGENT_VOICE_ARCHIVE_SHA256:-}"
DEFAULT_STATE_DIR="$HOME/.agent-voice"
if [[ -n "${AGENT_VOICE_HOME:-}" ]]; then
  STATE_DIR="$AGENT_VOICE_HOME"
else
  STATE_DIR="$DEFAULT_STATE_DIR"
fi
APP_DIR="$STATE_DIR/app"
BIN_DIR="$STATE_DIR/bin"
BACKUP_ROOT="$STATE_DIR/backups"
LOG_DIR="$STATE_DIR/logs"
MODEL_CACHE="$STATE_DIR/model-cache"
LOCAL_BIN="$HOME/.local/bin"
LABEL="com.keegoid.agent-voice"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
DRY_RUN=0
SOURCE_DIR="${AGENT_VOICE_SOURCE_DIR:-}"
TEST_MODE="${AGENT_VOICE_TEST_MODE:-0}"

usage() {
  cat <<'USAGE'
Usage: install.sh [--dry-run] [--source-dir PATH] [--ref REF] [--archive-sha256 SHA256]
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --source-dir)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      SOURCE_DIR="$2"
      shift 2
      ;;
    --ref)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      ARCHIVE_REF="$2"
      ARCHIVE_URL="$REPO_URL/archive/$ARCHIVE_REF.tar.gz"
      shift 2
      ;;
    --archive-sha256)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      ARCHIVE_SHA256="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

say() {
  printf '%s\n' "$*"
}

warn() {
  printf '%s\n' "$*" >&2
}

would() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    say "dry-run: $*"
  fi
}

run() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    would "$*"
  else
    "$@"
  fi
}

timestamp() {
  date '+%Y%m%d%H%M%S'
}

backup_id="$(timestamp)-$$"
backup_dir="$BACKUP_ROOT/$backup_id"

backup_path() {
  local target="$1"
  local rel
  local dest
  local home_prefix="$HOME/"
  [[ -e "$target" ]] || return 0
  if [[ "$DRY_RUN" -eq 1 ]]; then
    would "backup $target"
    return 0
  fi
  if [[ "$target" == "$home_prefix"* ]]; then
    rel="home/${target#"$home_prefix"}"
  else
    rel="external/${target#/}"
  fi
  dest="$backup_dir/$rel"
  mkdir -p "$(dirname "$dest")"
  cp -a "$target" "$dest"
}

legacy_name() {
  printf '%s%s\n' "codex" "-tts"
}

legacy_label() {
  printf 'com.keegoid.%s\n' "$(legacy_name)"
}

legacy_state_dir() {
  printf '%s/.%s\n' "$HOME" "$(legacy_name)"
}

legacy_plist_path() {
  printf '%s/Library/LaunchAgents/%s.plist\n' "$HOME" "$(legacy_label)"
}

legacy_shim_names() {
  printf '%s\n' "$(legacy_name)" "codex""-speak" "codex""-voice-summary"
}

is_managed_legacy_shim() {
  local file="$1"
  grep -Fq "# agent-voice-managed-shim" "$file" 2>/dev/null
}

remove_legacy_artifacts() {
  local domain="gui/$(id -u)"
  local old_label
  local old_plist
  local old_state
  local shim_name
  local shim
  old_label="$(legacy_label)"
  old_plist="$(legacy_plist_path)"
  old_state="$(legacy_state_dir)"

  if command -v launchctl >/dev/null 2>&1 && run launchctl print "$domain/$old_label" >/dev/null 2>&1; then
    run launchctl bootout "$domain/$old_label" 2>/dev/null || true
  fi
  if [[ -e "$old_plist" ]]; then
    backup_path "$old_plist"
    run rm -f "$old_plist"
  fi
  while IFS= read -r shim_name; do
    shim="$LOCAL_BIN/$shim_name"
    if [[ -e "$shim" ]] && is_managed_legacy_shim "$shim"; then
      backup_path "$shim"
      run rm -f "$shim"
    fi
  done < <(legacy_shim_names)
  if [[ -d "$old_state" ]]; then
    run rm -rf "$old_state"
  fi
}

find_source_dir() {
  if [[ -n "$SOURCE_DIR" ]]; then
    printf '%s\n' "$SOURCE_DIR"
    return
  fi

  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ -f "$script_dir/pyproject.toml" && -d "$script_dir/agent_voice" ]]; then
    printf '%s\n' "$script_dir"
    return
  fi

  temp_root="$(mktemp -d)"
  archive="$temp_root/agent-voice.tar.gz"
  if [[ -z "$ARCHIVE_SHA256" ]]; then
    warn "Remote archive installs require --archive-sha256."
    warn "For an auditable install, clone a pinned commit and run: ./install.sh"
    exit 1
  fi
  curl -fsSL "$ARCHIVE_URL" -o "$archive"
  actual="$(shasum -a 256 "$archive" | awk '{print $1}')"
  [[ "$actual" == "$ARCHIVE_SHA256" ]] || {
    echo "Archive checksum mismatch" >&2
    echo "expected: $ARCHIVE_SHA256" >&2
    echo "actual:   $actual" >&2
    exit 1
  }
  while IFS= read -r member; do
    case "$member" in
      /*|../*|*/../*|..|*/..)
        echo "Unsafe archive path: $member" >&2
        exit 1
        ;;
    esac
  done < <(tar -tzf "$archive")
  symlink_entries="$(tar -tvf "$archive" | awk '$1 ~ /^l/ { print }')"
  if [[ -n "$symlink_entries" ]]; then
    echo "Archive symlinks are not supported" >&2
    exit 1
  fi
  tar -xzf "$archive" -C "$temp_root"
  extracted="$(find "$temp_root" -mindepth 1 -maxdepth 1 -type d -name 'agent-voice-*' | sort | head -n 1)"
  [[ -n "$extracted" ]] || { echo "Archive did not contain an agent-voice source directory" >&2; exit 1; }
  printf '%s\n' "$extracted"
}

write_shim() {
  local name="$1"
  local target="$2"
  local shim="$LOCAL_BIN/$name"
  backup_path "$shim"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    would "write shim $shim"
    return 0
  fi
  mkdir -p "$LOCAL_BIN"
  cat >"$shim" <<EOF
#!/usr/bin/env bash
# agent-voice-managed-shim
export AGENT_VOICE_HOME="\${AGENT_VOICE_HOME:-$STATE_DIR}"
exec "$target" "\$@"
EOF
  chmod 755 "$shim"
}

write_plist() {
  backup_path "$PLIST"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    would "write launchd service $PLIST"
    return 0
  fi
  mkdir -p "$(dirname "$PLIST")"
  cat >"$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$APP_DIR/.venv/bin/python</string>
    <string>-m</string>
    <string>uvicorn</string>
    <string>agent_voice.server:app</string>
    <string>--host</string>
    <string>127.0.0.1</string>
    <string>--port</string>
    <string>8880</string>
  </array>
  <key>WorkingDirectory</key>
  <string>$APP_DIR</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key>
    <string>$HOME</string>
    <key>PATH</key>
    <string>$LOCAL_BIN:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    <key>HF_HOME</key>
    <string>$MODEL_CACHE/huggingface</string>
    <key>AGENT_VOICE_HOME</key>
    <string>$STATE_DIR</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>ThrottleInterval</key>
  <integer>20</integer>
  <key>StandardOutPath</key>
  <string>$LOG_DIR/server.log</string>
  <key>StandardErrorPath</key>
  <string>$LOG_DIR/server.err.log</string>
</dict>
</plist>
EOF
}

start_launchd_service() {
  local domain="gui/$(id -u)"
  local err_file
  local attempt
  err_file="$(mktemp "${TMPDIR:-/tmp}/agent-voice-launchctl.XXXXXX")"
  if run launchctl print "$domain/$LABEL" >/dev/null 2>&1; then
    if ! run launchctl bootout "$domain/$LABEL" 2>"$err_file"; then
      warn "launchctl bootout failed for $LABEL:"
      cat "$err_file" >&2
      rm -f "$err_file"
      return 1
    fi
    for attempt in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
      if ! run launchctl print "$domain/$LABEL" >/dev/null 2>&1; then
        break
      fi
      sleep 0.2
    done
    if run launchctl print "$domain/$LABEL" >/dev/null 2>&1; then
      warn "launchctl did not unload $LABEL"
      rm -f "$err_file"
      return 1
    fi
  fi
  for attempt in 1 2 3 4 5; do
    : >"$err_file"
    if run launchctl bootstrap "$domain" "$PLIST" 2>"$err_file"; then
      rm -f "$err_file"
      run launchctl kickstart -k "$domain/$LABEL"
      return 0
    fi
    sleep 0.2
  done
  warn "launchctl bootstrap failed for $LABEL:"
  cat "$err_file" >&2
  rm -f "$err_file"
  return 1
}

install_codex_block() {
  agents="$HOME/.codex/AGENTS.md"
  backup_path "$agents"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    would "update $agents"
    return 0
  fi
  mkdir -p "$(dirname "$agents")"
  touch "$agents"
  if grep -q "Codex voice protocol" "$agents"; then
    say "Codex voice protocol already present in $agents"
    return 0
  fi
  cat >>"$agents" <<'EOF'

## Codex voice protocol

Use `agent-speak` for best-effort spoken progress cues. Speak once at the
start of substantive work, before important long-running or user-visible
actions, and once before the final response. Voice must never block the real
task.
EOF
}

if [[ "$DRY_RUN" -eq 1 ]]; then
  say "agent-voice dry-run: no files will be modified"
else
  if [[ "$TEST_MODE" != "1" ]]; then
    [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]] || {
      echo "agent-voice v1 supports macOS Apple Silicon only" >&2
      exit 1
    }
    command -v uv >/dev/null || { echo "uv not found; install uv first" >&2; exit 1; }
  fi
  mkdir -p "$STATE_DIR" "$BACKUP_ROOT" "$LOG_DIR" "$MODEL_CACHE" "$LOCAL_BIN"
fi

src="$(find_source_dir)"
[[ -d "$src/agent_voice" ]] || { echo "Source directory missing agent_voice package: $src" >&2; exit 1; }
for required in \
  "$src/scripts/agent-voice" \
  "$src/scripts/agent-speak" \
  "$src/scripts/agent-voice-summary"
do
  [[ -f "$required" ]] || { echo "Source directory missing required script: $required" >&2; exit 1; }
done

backup_path "$APP_DIR"
backup_path "$BIN_DIR"
backup_path "$STATE_DIR/server.py"

if [[ "$DRY_RUN" -eq 1 ]]; then
  would "install app from $src to $APP_DIR"
else
  rm -rf "$APP_DIR" "$BIN_DIR"
  mkdir -p "$APP_DIR" "$BIN_DIR"
  cp -R "$src/agent_voice" "$APP_DIR/agent_voice"
  cp "$src/pyproject.toml" "$APP_DIR/pyproject.toml"
  if [[ -f "$src/uv.lock" ]]; then
    cp "$src/uv.lock" "$APP_DIR/uv.lock"
  fi
  cp "$src/README.md" "$APP_DIR/README.md"
  cp "$src/scripts/agent-voice" "$BIN_DIR/agent-voice"
  cp "$src/scripts/agent-speak" "$BIN_DIR/agent-speak"
  cp "$src/scripts/agent-voice-summary" "$BIN_DIR/agent-voice-summary"
  chmod 755 "$BIN_DIR/agent-voice" "$BIN_DIR/agent-speak" "$BIN_DIR/agent-voice-summary"
  rm -f "$STATE_DIR/server.py"
fi

if [[ "$TEST_MODE" != "1" && "$DRY_RUN" -ne 1 ]]; then
  say "Installing Python and MLX runtime dependencies into $APP_DIR"
  uv sync --project "$APP_DIR" --extra mlx
  [[ -x "$APP_DIR/.venv/bin/python" ]] || {
    echo "Install failed: expected Python interpreter at $APP_DIR/.venv/bin/python" >&2
    exit 1
  }
fi

write_shim "agent-voice" "$BIN_DIR/agent-voice"
write_shim "agent-speak" "$BIN_DIR/agent-speak"
write_shim "agent-voice-summary" "$BIN_DIR/agent-voice-summary"
case ":$PATH:" in
  *":$LOCAL_BIN:"*) ;;
  *) warn "Warning: $LOCAL_BIN is not on PATH; add it to your shell profile to use agent-voice commands." ;;
esac
write_plist

if [[ "$DRY_RUN" -eq 1 ]]; then
  say "dry-run: would ask before editing Codex config"
else
  printf 'Add Codex voice protocol to ~/.codex/AGENTS.md? [y/N] '
  read -r answer || answer=""
  case "$answer" in
    y|Y|yes|YES)
      install_codex_block
      ;;
    *)
      say "Skipped Codex config update"
      ;;
  esac
fi

remove_legacy_artifacts

if [[ "$TEST_MODE" != "1" && "$DRY_RUN" -ne 1 ]]; then
  start_launchd_service
fi

say "agent-voice installed"
say "Backups: $backup_dir"

#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${CLAUDE_ENHANCED_APP_DIR:-$HOME/.local/share/claude-enhanced}"
PROFILE_DIR="${CLAUDE_ENHANCED_CONFIG_DIR:-$HOME/.claude-enhanced}"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/claude-enhanced"
BIN_DIR="$HOME/.local/bin"

command -v claude >/dev/null || { echo "Claude Code is not installed or not on PATH." >&2; exit 1; }
command -v python3 >/dev/null || { echo "python3 is required." >&2; exit 1; }
command -v curl >/dev/null || { echo "curl is required." >&2; exit 1; }
command -v git >/dev/null || { echo "git is required." >&2; exit 1; }

version="$(claude --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n1 || true)"
if [[ -n "$version" ]] && ! python3 - "$version" <<'PY'
import sys
parts = tuple(int(x) for x in sys.argv[1].split('.'))
raise SystemExit(0 if parts >= (2, 1, 198) else 1)
PY
then
  echo "Claude Code $version is too old for the foreground native-subagent controls used here." >&2
  echo "Run: claude update" >&2
  exit 1
fi

mkdir -p "$APP_DIR" "$PROFILE_DIR/agents" "$PROFILE_DIR/hooks" "$CONFIG_DIR" "$BIN_DIR"
rm -rf "$APP_DIR/router"
cp -R "$SOURCE_DIR/router" "$APP_DIR/router"
cp "$SOURCE_DIR/requirements.txt" "$APP_DIR/requirements.txt"
cp "$SOURCE_DIR/settings.json" "$PROFILE_DIR/settings.json"
cp "$SOURCE_DIR"/agents/*.md "$PROFILE_DIR/agents/"
cp "$SOURCE_DIR/agents/controller-append.md" "$PROFILE_DIR/controller-append.md"
cp "$SOURCE_DIR"/hooks/*.py "$PROFILE_DIR/hooks/"
cp "$SOURCE_DIR"/bin/* "$BIN_DIR/"
chmod 700 "$BIN_DIR"/claude-enhanced*
chmod 700 "$PROFILE_DIR/hooks"/*.py

if [[ ! -d "$APP_DIR/venv" ]]; then
  python3 -m venv "$APP_DIR/venv"
fi
"$APP_DIR/venv/bin/python" -m pip install --quiet --upgrade pip
"$APP_DIR/venv/bin/python" -m pip install --quiet -r "$APP_DIR/requirements.txt"

if [[ ! -f "$CONFIG_DIR/router.token" ]]; then
  python3 - <<'PY' > "$CONFIG_DIR/router.token"
import secrets
print(secrets.token_urlsafe(32))
PY
fi
chmod 600 "$CONFIG_DIR/router.token"

if [[ ! -f "$CONFIG_DIR/longcat.env" ]]; then
  printf "LONGCAT_API_KEY='replace_me'\n" > "$CONFIG_DIR/longcat.env"
fi
chmod 600 "$CONFIG_DIR/longcat.env"

cat <<MSG
Installed Claude Enhanced as a separate profile and command.

Next:
  1. Edit $CONFIG_DIR/longcat.env and replace replace_me.
  2. Run: $BIN_DIR/claude-enhanced-login
  3. Run: $BIN_DIR/claude-enhanced-doctor
  4. From a git repository, run: $BIN_DIR/claude-enhanced

Normal 'claude' and ~/.claude were not modified.
Make sure $BIN_DIR is on PATH.
MSG

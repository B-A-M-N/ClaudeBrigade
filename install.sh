#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Canonical Brigade paths — accept old CLAUDE_ENHANCED_* as fallback for migration
APP_DIR="${CLAUDE_BRIGADE_APP_DIR:-${CLAUDE_ENHANCED_APP_DIR:-$HOME/.local/share/claude-brigade}}"
PROFILE_DIR="${CLAUDE_BRIGADE_PROFILE_DIR:-${CLAUDE_ENHANCED_CONFIG_DIR:-$HOME/.claude-brigade}}"
CONFIG_DIR="${BRIGADE_CONFIG_DIR:-${XDG_CONFIG_HOME:-$HOME/.config}/claude-brigade}"
STATE_DIR="${BRIGADE_STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/claude-brigade}"
CACHE_DIR="${BRIGADE_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/claude-brigade}"
BIN_DIR="$HOME/.local/bin"

# Migrate old ~/.config/claude-enhanced/config to new location if it exists and new does not
_OLD_CONFIG="${XDG_CONFIG_HOME:-$HOME/.config}/claude-enhanced"
if [[ -d "$_OLD_CONFIG" && ! -f "$CONFIG_DIR/models.yaml" ]]; then
  mkdir -p "$CONFIG_DIR"
  cp -n "$_OLD_CONFIG"/models.yaml "$CONFIG_DIR/" 2>/dev/null || true
  cp -n "$_OLD_CONFIG"/profiles.yaml "$CONFIG_DIR/" 2>/dev/null || true
  cp -n "$_OLD_CONFIG"/workflows.yaml "$CONFIG_DIR/" 2>/dev/null || true
  cp -n "$_OLD_CONFIG"/router.token "$CONFIG_DIR/" 2>/dev/null || true
  cp -n "$_OLD_CONFIG"/longcat.env "$CONFIG_DIR/" 2>/dev/null || true
fi

# Migrate old ~/.cache/claude-enhanced to new location
_OLD_CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/claude-enhanced"
if [[ -d "$_OLD_CACHE" && ! -d "$CACHE_DIR" ]]; then
  mv "$_OLD_CACHE" "$CACHE_DIR" 2>/dev/null || true
fi

# Migrate old ~/.local/state/claude-brigade was already created by earlier installs

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

mkdir -p "$APP_DIR" "$PROFILE_DIR" "$PROFILE_DIR/agents" "$PROFILE_DIR/hooks" "$CONFIG_DIR" "$BIN_DIR" \
         "$STATE_DIR" "$CACHE_DIR"

# Copy application code
rm -rf "$APP_DIR/router"
cp -R "$SOURCE_DIR/router" "$APP_DIR/router"
cp "$SOURCE_DIR/requirements.txt" "$APP_DIR/requirements.txt"
cp "$SOURCE_DIR/pyproject.toml" "$APP_DIR/pyproject.toml"

# Bundle the repo's default config/ alongside the installed code. This is
# NOT the operator's live config (that stays in $CONFIG_DIR and is never
# overwritten below) -- registry.py's get_registry() reads this bundled copy
# to backfill FreeInference/passthrough models an older, customized
# operator models.yaml predates (e.g. fastpath.yaml/sidecars.yaml referencing
# a model added to the repo defaults after the operator's config was last
# hand-edited). Without this directory present, that fallback silently finds
# nothing and registry validation fails outright instead of backfilling.
rm -rf "$APP_DIR/config"
cp -R "$SOURCE_DIR/config" "$APP_DIR/config"

# Copy settings to profile directory
cp "$SOURCE_DIR/settings.json" "$PROFILE_DIR/settings.json"

# Copy agent files
cp "$SOURCE_DIR"/agents/*.md "$PROFILE_DIR/agents/"
cp "$SOURCE_DIR/agents/controller-append.md" "$PROFILE_DIR/controller-append.md"

# Copy hooks
cp "$SOURCE_DIR"/hooks/*.py "$PROFILE_DIR/hooks/"

# Copy only declared executables to $BIN_DIR (not plan.md, etc.)
for _exe in claude-brigade claude-brigade-config claude-brigade-doctor claude-brigade-login claude-brigade-router-stop; do
  if [[ -f "$SOURCE_DIR/bin/$_exe" ]]; then
    cp "$SOURCE_DIR/bin/$_exe" "$BIN_DIR/$_exe"
    chmod 700 "$BIN_DIR/$_exe"
  fi
done

# Backward-compatibility symlinks — old names point to new binaries
ln -sf claude-brigade "$BIN_DIR/claude-enhanced"
ln -sf claude-brigade-doctor "$BIN_DIR/claude-enhanced-doctor"
ln -sf claude-brigade-login "$BIN_DIR/claude-enhanced-login"
ln -sf claude-brigade-router-stop "$BIN_DIR/claude-enhanced-router-stop"
chmod 700 "$PROFILE_DIR/hooks"/*.py

# Set up Python virtual environment
if [[ ! -d "$APP_DIR/venv" ]]; then
  python3 -m venv "$APP_DIR/venv"
fi
"$APP_DIR/venv/bin/python" -m pip install --quiet --upgrade pip
"$APP_DIR/venv/bin/python" -m pip install --quiet -r "$APP_DIR/requirements.txt"

# Install the enhanced_router package in editable mode so hooks can
# ``import enhanced_router.state`` without PYTHONPATH tricks.
"$APP_DIR/venv/bin/python" -m pip install --quiet -e "$APP_DIR"

# --- Config directory management -------------------------------------------
CONFIG_MIGRATED="${CACHE_DIR}/config_migrated_v1"

if [[ ! -f "$CONFIG_MIGRATED" ]]; then
  # ---- First install: create default config files -------------------------
  if [[ ! -f "$CONFIG_DIR/models.yaml" ]]; then
    cp "$SOURCE_DIR/config/models.yaml" "$CONFIG_DIR/models.yaml"
  fi
  if [[ ! -f "$CONFIG_DIR/profiles.yaml" ]]; then
    cp "$SOURCE_DIR/config/profiles.yaml" "$CONFIG_DIR/profiles.yaml"
  fi
  if [[ ! -f "$CONFIG_DIR/workflows.yaml" ]]; then
    cp "$SOURCE_DIR/config/workflows.yaml" "$CONFIG_DIR/workflows.yaml"
  fi
  if [[ ! -f "$CONFIG_DIR/providers.yaml" ]]; then
    cp "$SOURCE_DIR/config/providers.yaml" "$CONFIG_DIR/providers.yaml"
  fi
  if [[ ! -f "$CONFIG_DIR/fastpath.yaml" ]]; then
    cp "$SOURCE_DIR/config/fastpath.yaml" "$CONFIG_DIR/fastpath.yaml"
  fi
  if [[ ! -f "$CONFIG_DIR/sidecars.yaml" ]]; then
    cp "$SOURCE_DIR/config/sidecars.yaml" "$CONFIG_DIR/sidecars.yaml"
  fi

  # Generate router token (only on first install)
  if [[ ! -f "$CONFIG_DIR/router.token" ]]; then
    python3 -c "
import os, sys, secrets
path = sys.argv[1]
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_TRUNC
if hasattr(os, 'O_NOFOLLOW'):
    flags |= os.O_NOFOLLOW
if os.path.islink(path):
    print('ERROR: router.token is a symlink', file=sys.stderr)
    sys.exit(1)
fd = os.open(path, flags, 0o600)
with os.fdopen(fd, 'w') as fh:
    fh.write(secrets.token_urlsafe(32))
os.chmod(path, 0o600)
" "$CONFIG_DIR/router.token"
  fi
  chmod 600 "$CONFIG_DIR/router.token"

  # Generate default providers.env (replaces longcat.env)
  if [[ ! -f "$CONFIG_DIR/providers.env" ]]; then
    cat > "$CONFIG_DIR/providers.env" <<'ENV'
# Provider API keys — one KEY=VALUE per line.
# Only recognized variable names are loaded.
# FREEINFERENCE_API_KEY=
# FREEINFERENCE_API_BASE=
# FREEINFERENCE_OPENAI_BASE=
# FREEINFERENCE_ANTHROPIC_BASE=
# FREEINFERENCE_MAX_CONCURRENCY=4
# LONGCAT_API_KEY=
# OPENROUTER_API_KEY=
# DEEPSEEK_API_KEY=
# KILO_API_KEY=
# CLINE_API_KEY=
# CLINEPASS_API_KEY=
# OPENCODE_API_KEY=
# OPENCODEGO_API_KEY=
# NVIDIA_API_KEY=
# FREETHEAI_API_KEY=
# REQUESTY_API_KEY=
# FREEMODEL_API_KEY=
ENV
  fi
  chmod 600 "$CONFIG_DIR/providers.env"

  # Generate LiteLLM internal key
  if [[ ! -f "$CONFIG_DIR/litellm.token" ]]; then
    python3 -c "
import os, sys, secrets
path = sys.argv[1]
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_TRUNC
if hasattr(os, 'O_NOFOLLOW'):
    flags |= os.O_NOFOLLOW
if os.path.islink(path):
    print('ERROR: litellm.token is a symlink', file=sys.stderr)
    sys.exit(1)
fd = os.open(path, flags, 0o600)
with os.fdopen(fd, 'w') as fh:
    fh.write(secrets.token_urlsafe(32))
os.chmod(path, 0o600)
" "$CONFIG_DIR/litellm.token"
  fi
  chmod 600 "$CONFIG_DIR/litellm.token"

  # Migrate old longcat.env if new providers.env doesn't exist yet
  _OLD_LC="${CONFIG_DIR}/longcat.env"
  if [[ -f "$_OLD_LC" ]] && [[ ! -f "$CONFIG_DIR/providers.env" ]]; then
    if grep -q 'LONGCAT_API_KEY=' "$_OLD_LC" && ! grep -q 'replace_me' "$_OLD_LC"; then
      grep '^LONGCAT_API_KEY=' "$_OLD_LC" > "$CONFIG_DIR/providers.env" 2>/dev/null || true
    fi
  fi

  touch "$CONFIG_MIGRATED"
else
  # ---- Upgrade path: install new defaults as .example files ---------------
  for _default in models.yaml profiles.yaml workflows.yaml providers.yaml fastpath.yaml sidecars.yaml; do
    if [[ -f "$CONFIG_DIR/$_default" ]]; then
      cp "$SOURCE_DIR/config/$_default" "$CONFIG_DIR/${_default}.example"
    else
      cp "$SOURCE_DIR/config/$_default" "$CONFIG_DIR/$_default"
    fi
  done

  # Router token: generate only if missing
  if [[ ! -f "$CONFIG_DIR/router.token" ]]; then
    python3 - <<'PY' > "$CONFIG_DIR/router.token"
import secrets
print(secrets.token_urlsafe(32))
PY
    chmod 600 "$CONFIG_DIR/router.token"
  fi

  # providers.env: generate only if missing
  if [[ ! -f "$CONFIG_DIR/providers.env" ]]; then
    # Migrate from longcat.env if available
    if [[ -f "$CONFIG_DIR/longcat.env" ]]; then
      if grep -q 'LONGCAT_API_KEY=' "$CONFIG_DIR/longcat.env" && ! grep -q 'replace_me' "$CONFIG_DIR/longcat.env"; then
        grep '^LONGCAT_API_KEY=' "$CONFIG_DIR/longcat.env" > "$CONFIG_DIR/providers.env" 2>/dev/null || true
      fi
    fi
    if [[ ! -f "$CONFIG_DIR/providers.env" ]]; then
      cat > "$CONFIG_DIR/providers.env" <<'ENV'
# Provider API keys — one KEY=VALUE per line.
# Only recognized variable names are loaded.
# FREEINFERENCE_API_KEY=
# FREEINFERENCE_API_BASE=
# FREEINFERENCE_OPENAI_BASE=
# FREEINFERENCE_ANTHROPIC_BASE=
# FREEINFERENCE_MAX_CONCURRENCY=4
# LONGCAT_API_KEY=
# OPENROUTER_API_KEY=
# KILO_API_KEY=
# CLINE_API_KEY=
# CLINEPASS_API_KEY=
# OPENCODE_API_KEY=
# OPENCODEGO_API_KEY=
# NVIDIA_API_KEY=
# FREETHEAI_API_KEY=
# REQUESTY_API_KEY=
# FREEMODEL_API_KEY=
ENV
    fi
    chmod 600 "$CONFIG_DIR/providers.env"
  fi

  # litellm.token: generate only if missing
  if [[ ! -f "$CONFIG_DIR/litellm.token" ]]; then
    python3 - <<'PY' > "$CONFIG_DIR/litellm.token"
import secrets
print(secrets.token_urlsafe(32))
PY
    chmod 600 "$CONFIG_DIR/litellm.token"
  fi
fi

# ---- Clean up old managed agent files --------------------------------------
# These were part of the previous agent naming convention.
# Agents live under "$PROFILE_DIR/agents/", not "$PROFILE_DIR/" directly.
for _old in enhanced-controller.md longcat-recon.md longcat-implementer.md longcat-adversary.md longcat-repairer.md; do
  rm -f "$PROFILE_DIR/agents/$_old"
done

cat <<MSG
Installed ClaudeBrigade as a separate profile and command.

Next:
  1. Run: $BIN_DIR/claude-brigade-config  (paste keys securely)
  2. Run: $BIN_DIR/claude-brigade-login
  3. Run: $BIN_DIR/claude-brigade-doctor
  4. From a git repository, run: $BIN_DIR/claude-brigade

Normal 'claude' and ~/.claude were not modified.
Make sure \$BIN_DIR is on PATH.
MSG

# Offer immediate interactive setup after installation.  The prompt is only
# shown for a real terminal so scripted/headless installs remain usable.
# Set CLAUDE_BRIGADE_SKIP_CONFIG=1 to suppress it explicitly.
if [[ -t 0 && -t 1 && "${CLAUDE_BRIGADE_SKIP_CONFIG:-0}" != "1" ]]; then
  printf '\nConfigure provider keys, sidecars, and inference profiles now? [Y/n] '
  read -r _configure_now || _configure_now="n"
  if [[ -z "$_configure_now" || "$_configure_now" =~ ^[Yy]$ ]]; then
    exec "$BIN_DIR/claude-brigade-config"
  fi
fi

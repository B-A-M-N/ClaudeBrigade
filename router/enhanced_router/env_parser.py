"""Safe, nonexecuting dotenv parser with strict validation."""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path


class EnvParseError(ValueError):
    """Raised when a providers.env file is invalid."""


# Regex for a valid shell / env variable name (POSIX-compatible).
_VALID_VAR_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Set of common credential / API variable names that may appear in
# providers.env.  The parser accepts ANY name matching the regex when
# *allowed_keys* is not provided, and rejects names outside the set when
# it is provided.


def parse_env_file(
    path: str | Path,
    allowed_keys: set[str] | frozenset[str] | list[str] | tuple[str, ...] | None = None,
) -> dict[str, str]:
    """Parse a dotenv file with strict validation.

    Args:
        path: Path to the env file.
        allowed_keys: If set, only these variable names will be returned.
                      Keys outside this set raise EnvParseError.

    Returns:
        Dict of {variable_name: value}.

    Raises:
        EnvParseError: On any validation failure.
    """
    path = Path(path)

    # ------------------------------------------------------------------
    # 1. O_NOFOLLOW — refuse symlinks entirely
    # ------------------------------------------------------------------
    if path.is_symlink():
        raise EnvParseError("Refusing to parse symlinked env file")

    # ------------------------------------------------------------------
    # 2. Validate regular file ownership and mode
    # ------------------------------------------------------------------
    try:
        st = path.lstat()
    except OSError as exc:
        raise EnvParseError(f"Cannot stat env file: {exc}") from exc

    # Must be a regular file (not a directory, socket, etc.)
    if not stat.S_ISREG(st.st_mode):
        raise EnvParseError("Env file is not a regular file")

    # Must be owned by the current user or root
    uid = st.st_uid
    effective_uid = os.geteuid()
    if uid not in (os.getuid(), effective_uid, 0):
        raise EnvParseError(
            f"Env file owned by uid={uid}, current uid={os.getuid()}; "
            "refusing to parse"
        )

    # Must not be writable or executable by group or others
    mode = stat.S_IMODE(st.st_mode)
    if mode & (stat.S_IWGRP | stat.S_IXGRP | stat.S_IWOTH | stat.S_IXOTH):
        raise EnvParseError(
            f"Env file has excessive permissions: {oct(mode)}"
        )

    # ------------------------------------------------------------------
    # 3. Read file content — reject null bytes
    # ------------------------------------------------------------------
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise EnvParseError(f"Cannot read env file: {exc}") from exc

    if b"\x00" in raw:
        raise EnvParseError("Env file contains null bytes")

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EnvParseError(f"Env file is not valid UTF-8: {exc}") from exc

    # ------------------------------------------------------------------
    # 4. Parse lines with quoted-value support
    # ------------------------------------------------------------------
    result: dict[str, str] = {}
    allowed_set: set[str] | None = None
    if allowed_keys is not None:
        allowed_set = set(allowed_keys)

    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()

        # Skip blank lines and comments
        if not stripped or stripped.startswith("#"):
            continue

        # Must contain '='
        eq_idx = stripped.find("=")
        if eq_idx < 1:
            raise EnvParseError(
                f"Line {lineno}: expected KEY=VALUE, got {stripped!r}"
            )

        key = stripped[:eq_idx].strip()
        value = stripped[eq_idx + 1:]

        # Validate variable name
        if not _VALID_VAR_RE.match(key):
            raise EnvParseError(
                f"Line {lineno}: invalid variable name {key!r}"
            )

        # Reject duplicate keys
        if key in result:
            raise EnvParseError(
                f"Line {lineno}: duplicate key {key!r}"
            )

        # Parse value — handle quoting
        value = _parse_value(value)

        # If allowed_keys was provided, reject unapproved names
        if allowed_set is not None and key not in allowed_set:
            raise EnvParseError(
                f"Line {lineno}: unapproved key {key!r}"
            )

        result[key] = value

    return result


def _parse_value(raw: str) -> str:
    """Extract a value from the right-hand side of KEY=VALUE.

    Supports:
      - Unquoted values (trailing whitespace is trimmed)
      - Double-quoted values with escape support
      - Single-quoted values (no escape processing)

    Returns the raw parsed string value.
    """
    stripped = raw.strip()

    if not stripped:
        return ""

    # Double-quoted
    if stripped.startswith('"'):
        inner = _unquote_double(stripped[1:])
        if inner is not None:
            return inner
        # Unclosed quote — treat rest as unquoted
        return stripped[1:].strip()

    # Single-quoted (no escape processing)
    if stripped.startswith("'"):
        end = stripped.find("'", 1)
        if end < 0:
            # Unclosed quote — treat rest as unquoted
            return stripped[1:].strip()
        return stripped[1:end]

    # Unquoted — trim trailing whitespace/comments
    # (basic: strip trailing whitespace; inline comments not supported
    #  by strict mode to avoid ambiguity)
    return stripped.rstrip()


def _unquote_double(remaining: str) -> str | None:
    """Extract content inside double quotes, handling escape sequences.

    Returns the unquoted content, or None if the closing quote is not found.
    """
    result: list[str] = []
    i = 0
    while i < len(remaining):
        ch = remaining[i]
        if ch == '\\':
            # Escape sequence
            if i + 1 < len(remaining):
                next_ch = remaining[i + 1]
                if next_ch == '"':
                    result.append('"')
                    i += 2
                    continue
                elif next_ch == '\\':
                    result.append('\\')
                    i += 2
                    continue
            # Unknown escape — keep as-is
            result.append(ch)
            i += 1
        elif ch == '"':
            # Found closing quote
            return "".join(result)
        else:
            result.append(ch)
            i += 1
    return None  # No closing quote found

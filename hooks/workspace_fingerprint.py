#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import os
import pathlib
import subprocess
import sys

# Cache fingerprint results for the duration of a session.
# Keyed by the epoch ID to ensure cache invalidation on epoch boundaries.
_fingerprint_cache: dict[str, str] = {}


def run(root: pathlib.Path, *args: str, text: bool = False) -> bytes | str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args],
        stderr=subprocess.DEVNULL,
        text=text,
    )


def repository_root(cwd: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(str(run(cwd, "rev-parse", "--show-toplevel", text=True)).strip())


def git_has_head(root: pathlib.Path) -> bool:
    try:
        run(root, "rev-parse", "--verify", "HEAD")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _get_epoch_id(session_id: str | None = None) -> str | None:
    """Get the current epoch ID from the Claude session directory."""
    cache = pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")) / "claude-brigade"
    session = session_id or os.environ.get("CLAUDE_BRIGADE_SESSION_ID", "")
    if not session:
        return None
    epoch_file = cache / "sessions" / session / "active_epoch_id.txt"
    try:
        return epoch_file.read_text(encoding="utf-8").strip()
    except (OSError, FileNotFoundError):
        return None


def fingerprint(
    cwd: pathlib.Path,
    *,
    session_id: str | None = None,
    epoch_id: str | None = None,
) -> str:
    """Compute SHA-256 fingerprint of tracked changes and untracked files.

    Results are cached per epoch to avoid redundant git subprocess calls.
    Cache is invalidated when the epoch changes.
    """
    root = repository_root(cwd)

    # Check cache with epoch-based invalidation
    active_epoch = epoch_id or _get_epoch_id(session_id)
    cache_key = f"{root}:{active_epoch}" if active_epoch else None
    if cache_key and cache_key in _fingerprint_cache:
        return _fingerprint_cache[cache_key]

    digest = hashlib.sha256()
    digest.update(b"tracked-diff\0")
    if git_has_head(root):
        diff_output = run(root, "diff", "--binary", "--no-ext-diff", "HEAD")
        if isinstance(diff_output, bytes):
            digest.update(diff_output)
        else:
            digest.update(diff_output.encode("utf-8"))
    else:
        diff_output = run(root, "diff", "--binary", "--no-ext-diff")
        if isinstance(diff_output, bytes):
            digest.update(diff_output)
        else:
            digest.update(diff_output.encode("utf-8"))
        diff_output2 = run(root, "diff", "--binary", "--no-ext-diff", "--staged")
        if isinstance(diff_output2, bytes):
            digest.update(diff_output2)
        else:
            digest.update(diff_output2.encode("utf-8"))

    raw = run(root, "ls-files", "--others", "--exclude-standard", "-z")
    assert isinstance(raw, bytes)
    for encoded in sorted(part for part in raw.split(b"\0") if part):
        relative = encoded.decode("utf-8", errors="surrogateescape")
        path = root / relative
        digest.update(b"untracked\0")
        digest.update(encoded)
        digest.update(b"\0")
        if path.is_symlink():
            digest.update(b"symlink\0")
            digest.update(path.readlink().as_posix().encode())
        elif path.is_file():
            digest.update(path.read_bytes())
        else:
            digest.update(b"non-file")
        digest.update(b"\0")

    result = digest.hexdigest()

    # Cache the result
    if cache_key:
        _fingerprint_cache[cache_key] = result

    return result


def clear_fingerprint_cache() -> None:
    """Clear the fingerprint cache. Called during epoch transitions."""
    _fingerprint_cache.clear()


def main() -> int:
    cwd = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    try:
        print(fingerprint(cwd))
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("not-a-git-repository", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

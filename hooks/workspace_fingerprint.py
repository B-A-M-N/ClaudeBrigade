#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import pathlib
import subprocess
import sys

# All git calls in this file route through here, so a stalled git process
# (e.g. a lock held by another process, a huge repo on a slow filesystem)
# can't hang a fingerprint computation indefinitely -- callers should treat
# subprocess.TimeoutExpired the same as any other fingerprint failure.
_GIT_TIMEOUT_SECONDS = 4.0


def run(root: pathlib.Path, *args: str, text: bool = False) -> bytes | str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args],
        stderr=subprocess.DEVNULL,
        text=text,
        timeout=_GIT_TIMEOUT_SECONDS,
    )


def repository_root(cwd: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(str(run(cwd, "rev-parse", "--show-toplevel", text=True)).strip())


def git_has_head(root: pathlib.Path) -> bool:
    try:
        run(root, "rev-parse", "--verify", "HEAD")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def fingerprint(
    cwd: pathlib.Path,
    *,
    session_id: str | None = None,
    epoch_id: str | None = None,
) -> str:
    """Compute SHA-256 fingerprint of tracked changes and untracked files.

    This function intentionally does not cache mutable workspace state. Hooks
    compute pre/post fingerprints in the same epoch, and a cache keyed only by
    epoch would make every mutation look unchanged.
    """
    root = repository_root(cwd)

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
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
        else:
            digest.update(b"non-file")
        digest.update(b"\0")

    result = digest.hexdigest()

    return result


def clear_fingerprint_cache() -> None:
    """Compatibility no-op for callers from the old cached implementation."""


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

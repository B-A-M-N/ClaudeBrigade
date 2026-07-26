#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import pathlib
import subprocess
import sys


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


def fingerprint(cwd: pathlib.Path) -> str:
    root = repository_root(cwd)
    digest = hashlib.sha256()
    digest.update(b"tracked-diff\0")
    if git_has_head(root):
        digest.update(run(root, "diff", "--binary", "--no-ext-diff", "HEAD"))
    else:
        digest.update(run(root, "diff", "--binary", "--no-ext-diff"))
        digest.update(run(root, "diff", "--binary", "--no-ext-diff", "--staged"))

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
    return digest.hexdigest()


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

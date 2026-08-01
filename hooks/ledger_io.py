"""Small cross-process-safe JSONL helpers for Claude Code hook evidence."""

from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def _locked(path: Path, *, shared: bool = False) -> Iterator[None]:
    """Hold a per-log POSIX lock while reading or appending evidence."""
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def append_jsonl(path: Path, record: dict) -> None:
    """Append one complete, fsynced JSON record under an inter-process lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(record, separators=(",", ":")) + "\n"
    with _locked(path):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict]:
    """Read complete JSON objects while sharing the writer's lock."""
    if not path.exists():
        return []
    try:
        with _locked(path, shared=True):
            lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    records: list[dict] = []
    for line in lines:
        try:
            record = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(record, dict):
            records.append(record)
    return records

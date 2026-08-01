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


def read_jsonl_cached(path: Path) -> list[dict]:
    """Like read_jsonl, but tails from a remembered byte offset instead of
    re-reading the whole file, using a sidecar cache file next to *path*
    (e.g. ledger.jsonl -> ledger.jsonl.cache.json) that stores {mtime, size,
    offset, records}. Falls back to a full re-read when the sidecar is
    missing/corrupt, or when the source file's mtime doesn't match the
    cache (covers rotation/truncation, not just growth) or its size is
    smaller than the cached offset (covers truncation even with a stale
    mtime on some filesystems). Must apply the exact same malformed-line
    skipping behavior as read_jsonl -- never raise on a bad line, never
    silently drop a good line. Must be safe to call concurrently with
    append_jsonl (which already takes a flock) -- take a *shared* lock
    (not exclusive) while reading, matching whatever locking primitive
    ledger_io.py already uses for read_jsonl.
    """
    cache_path = path.with_name(path.name + ".cache.json")

    if not path.exists():
        return []

    # Attempt to load the cache
    cache_state = None
    if cache_path.exists():
        try:
            cache_data = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cache_data, dict) and all(k in cache_data for k in ["mtime", "size", "offset", "records"]):
                cache_state = cache_data
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    # Get current file stats
    try:
        stat = path.stat()
        current_mtime = stat.st_mtime
        current_size = stat.st_size
    except OSError:
        return []

    # Determine if we can use the cache
    if cache_state is not None:
        cache_mtime = cache_state.get("mtime")
        cache_size = cache_state.get("size")
        cache_offset = cache_state.get("offset")
        cached_records = cache_state.get("records", [])

        # Cache is valid if: mtime matches and size >= cached offset
        # (mtime mismatch covers rotation/truncation; size < offset covers truncation)
        if (cache_mtime == current_mtime and
            current_size >= cache_offset):
            # Use cache and tail from offset
            tail_records = _read_jsonl_tail(path, cache_offset)
            combined = cached_records + tail_records

            # Update cache with new tail
            try:
                with _locked(cache_path):
                    new_cache = {
                        "mtime": current_mtime,
                        "size": current_size,
                        "offset": current_size,
                        "records": combined,
                    }
                    cache_path.write_text(json.dumps(new_cache, separators=(",", ":")), encoding="utf-8")
            except OSError:
                pass

            return combined

    # Cache miss or invalid: full re-read
    records = read_jsonl(path)

    # Write new cache
    try:
        with _locked(cache_path):
            new_cache = {
                "mtime": current_mtime,
                "size": current_size,
                "offset": current_size,
                "records": records,
            }
            cache_path.write_text(json.dumps(new_cache, separators=(",", ":")), encoding="utf-8")
    except OSError:
        pass

    return records


def _read_jsonl_tail(path: Path, offset: int) -> list[dict]:
    """Read JSON lines from a specific byte offset to end of file,
    applying the same malformed-line skipping as read_jsonl.
    """
    try:
        with _locked(path, shared=True):
            with path.open("rb") as f:
                f.seek(offset)
                remaining = f.read()
        text = remaining.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    records: list[dict] = []
    for line in text.splitlines():
        if not line:  # Skip empty lines from the splitlines() at the boundary
            continue
        try:
            record = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(record, dict):
            records.append(record)
    return records

"""File-based process lock using flock(2).

Provides cross-process mutual exclusion through POSIX file locks,
used to prevent concurrent launcher invocations from racing on
port selection, PID file creation, and token generation.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
from types import TracebackType


class FileLock:
    """A file-based lock using flock(2) for cross-process mutual exclusion."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._fd: int | None = None

    def acquire(self, blocking: bool = True) -> bool:
        """Try to acquire the lock. Returns False if non-blocking and unavailable."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(str(self._path), os.O_CREAT, 0o600)
        try:
            op = fcntl.LOCK_EX | fcntl.LOCK_NB
            if blocking:
                op = fcntl.LOCK_EX
            fcntl.flock(self._fd, op)
            return True
        except (BlockingIOError, OSError):
            os.close(self._fd)
            self._fd = None
            return False

    def release(self) -> None:
        """Release the lock."""
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    def __enter__(self) -> FileLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.release()

"""Tests for router/enhanced_router/lock.py — FileLock."""

from __future__ import annotations

import os
import stat
from pathlib import Path


from enhanced_router.lock import FileLock


class TestFileLock:
    """Tests for the flock-based FileLock class."""

    def test_acquire_and_release(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "test.lock"
        lock = FileLock(lock_path)
        assert lock.acquire() is True
        lock.release()
        # After release it should be re-acquirable
        assert lock.acquire() is True
        lock.release()

    def test_acquire_with_context_manager(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "ctx.lock"
        with FileLock(lock_path) as lock:
            # While inside the context, the fd should exist
            assert lock._fd is not None
        # After exiting, fd should be cleared
        assert lock._fd is None

    def test_non_blocking_fails_when_held(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "contention.lock"
        lock_a = FileLock(lock_path)
        lock_b = FileLock(lock_path)
        assert lock_a.acquire() is True
        # lock_b should fail in non-blocking mode
        assert lock_b.acquire(blocking=False) is False
        lock_a.release()
        lock_b.release()

    def test_acquire_twice_different_instances(self, tmp_path: Path) -> None:
        """Two FileLock instances on the same file: second non-blocking acquire fails."""
        lock_path = tmp_path / "dual.lock"
        lock_a = FileLock(lock_path)
        lock_b = FileLock(lock_path)
        assert lock_a.acquire() is True
        # lock_b should fail in non-blocking mode
        assert lock_b.acquire(blocking=False) is False
        lock_a.release()
        lock_b.release()

    def test_file_permissions(self, tmp_path: Path) -> None:
        """Lock file should be created with restrictive permissions."""
        lock_path = tmp_path / "perm.lock"
        lock = FileLock(lock_path)
        lock.acquire()
        lock.release()
        mode = os.stat(lock_path).st_mode & 0o777
        assert mode == stat.S_IRUSR | stat.S_IWUSR  # 0o600

    def test_parent_dir_created(self, tmp_path: Path) -> None:
        """FileLock creates parent directories if needed."""
        nested = tmp_path / "sub" / "deep" / "lockfile"
        lock = FileLock(nested)
        assert lock.acquire() is True
        assert nested.exists()
        lock.release()

    def test_double_release_is_safe(self, tmp_path: Path) -> None:
        """Calling release() twice should not raise."""
        lock = FileLock(tmp_path / "dr.lock")
        lock.acquire()
        lock.release()
        lock.release()  # should be a no-op

    def test_context_manager_double_exit_is_safe(self, tmp_path: Path) -> None:
        lock = FileLock(tmp_path / "cm.lock")
        lock.__enter__()
        lock.__exit__(None, None, None)
        lock.__exit__(None, None, None)  # should not raise

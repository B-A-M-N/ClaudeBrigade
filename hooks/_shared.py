"""Helpers shared across hooks/*.py.

Kept dependency-free (stdlib + ledger_io, no enhanced_router import) so
every hook can use it without pulling in the router package just to run
PreToolUse/PostToolUse.
"""
from __future__ import annotations

import os
import pathlib
import sys
from typing import Callable

from ledger_io import append_jsonl


def record_ledger(session_dir: pathlib.Path, record: dict) -> None:
    """Append one event to this session's ledger.jsonl.

    Previously reimplemented identically in both audit_tool.py and
    audit_agent.py.
    """
    append_jsonl(session_dir / "ledger.jsonl", record)


def resolve_session_dir(data: dict) -> pathlib.Path:
    """Return this hook invocation's session_dir, without creating it --
    callers that need the directory (or an "active" subdirectory) to exist
    still call .mkdir() themselves, since which subdirectories are needed
    differs per hook.

    Previously reimplemented identically in both audit_tool.py and
    audit_agent.py.
    """
    cache = pathlib.Path(os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")) / "claude-brigade"
    session = str(data.get("session_id", "unknown"))
    return cache / "sessions" / session


def read_active_epoch_id(session_dir: pathlib.Path) -> str:
    """Return the active epoch_id marker, or "ep_unknown" if none is set yet.

    Previously reimplemented identically in both audit_tool.py and
    audit_agent.py.
    """
    epoch_file = session_dir / "active_epoch_id.txt"
    return epoch_file.read_text(encoding="utf-8").strip() if epoch_file.exists() else "ep_unknown"


def fail_open_main(main: Callable[[], int]) -> int:
    """Run an observational hook's main(), never letting it crash uncaught.

    guard_tool.py and completion_guard.py have real enforcement teeth (they
    can deny/block) and must NOT use this -- a bug there should be loud, not
    silently swallowed. Every other hook only records evidence; an unhandled
    exception there shouldn't turn into a crashed subprocess (a traceback on
    stderr, a nonzero exit) that could interrupt the tool call it's merely
    trying to observe. Logs the exception to stderr and exits 0 either way.
    """
    try:
        return main()
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
        sys.stderr.write(f"{main.__module__}: unhandled exception, failing open: {exc}\n")
        return 0

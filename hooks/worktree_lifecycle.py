#!/usr/bin/env python3
"""Claude Code WorktreeCreate/WorktreeRemove authority.

WorktreeCreate replaces Claude Code's default Git setup so native mutators
receive the same dirty/untracked baseline that ClaudeBrigade records for the
canonical checkout.  WorktreeRemove is observational: Claude Code remains the
owner of removal, while the Brigade audit records the lifecycle boundary.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _shared import resolve_session_dir  # noqa: E402
from ledger_io import append_jsonl  # noqa: E402


def _read() -> dict:
    payload = json.load(sys.stdin)
    if not isinstance(payload, dict):
        raise ValueError("Worktree hook payload must be an object")
    return payload


def main() -> int:
    try:
        payload = _read()
        event = str(payload.get("hook_event_name") or "")
        session_dir = resolve_session_dir(payload)
        session_dir.mkdir(parents=True, exist_ok=True)
        record: dict[str, Any] = {
            "event": event,
            "session_id": str(payload.get("session_id") or "unknown"),
            "run_id": os.environ.get("CLAUDE_BRIGADE_RUN_ID", ""),
            "name": str(payload.get("name") or ""),
            "cwd": str(payload.get("cwd") or ""),
            "worktree_path": str(payload.get("worktree_path") or ""),
        }
        if event == "WorktreeCreate":
            from enhanced_router.shadow_worktree import ShadowWorktreeManager

            cwd = pathlib.Path(record["cwd"] or ".").resolve()
            manager = ShadowWorktreeManager(cwd)
            run_id = record["run_id"] or "standalone"
            path, baseline = manager.create_preserved_native_worktree(
                run_id=run_id,
                name=record["name"] or "native",
            )
            record["worktree_path"] = str(path)
            record["base_sha"] = baseline.base_sha
            record["dirty_patch_hash"] = baseline.dirty_patch_hash
            record["untracked_files"] = list(baseline.untracked_files)
            append_jsonl(session_dir / "worktrees.jsonl", record)
            # stdout is the WorktreeCreate protocol; all diagnostics stay on
            # stderr so Claude Code receives only the absolute path.
            print(path)
            return 0
        if event == "WorktreeRemove":
            append_jsonl(session_dir / "worktrees.jsonl", record)
            return 0
        print(f"unsupported worktree hook event: {event}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ClaudeBrigade worktree hook failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

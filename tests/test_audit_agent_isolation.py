"""Tests for hooks/audit_agent.py's _execution_workspace isolation checks.

This is the actual production-wired isolation validator (registered by
SubagentStart, hooks/audit_agent.py:main) -- shadow_workspace_state.py's
validate_execution_workspace was a separate, never-called DB-only check
removed as dead code. These tests exercise the real rejection paths, which
had no direct coverage before.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from audit_agent import WorkspaceIsolationError, _execution_workspace
from enhanced_router.shadow_worktree import ShadowWorktreeManager
from enhanced_router.state import RouteState


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode:
        raise AssertionError(result.stderr.decode())
    return result.stdout.decode().strip()


@pytest.fixture()
def git_repo(tmp_path: Path) -> tuple[Path, RouteState, ShadowWorktreeManager]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "tests@example.invalid")
    _git(repo, "config", "user.name", "ClaudeBrigade Tests")
    (repo / "tracked.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-qm", "base")
    state = RouteState(tmp_path / "state.db")
    state.create_run("run-1", session_id="session-1", cwd=str(repo))
    state.create_epoch("run-1", "epoch-1", "normal", "profile-1")
    manager = ShadowWorktreeManager(repo, worktree_root=tmp_path / "shadows")
    return repo, state, manager


def test_execution_workspace_returns_none_for_a_non_mutating_role(
    git_repo: tuple[Path, RouteState, ShadowWorktreeManager],
):
    repo, state, _manager = git_repo
    result = _execution_workspace(
        state, "run-1", "epoch-1", "exec-1", repo, "brigade-recon",
    )
    assert result is None


def test_execution_workspace_rejects_a_mutating_agent_in_the_canonical_checkout(
    git_repo: tuple[Path, RouteState, ShadowWorktreeManager],
):
    """A mutating native agent resolving to the main checkout instead of an
    isolated worktree must be rejected outright."""
    repo, state, _manager = git_repo
    with pytest.raises(WorkspaceIsolationError, match="canonical checkout"):
        _execution_workspace(state, "run-1", "epoch-1", "exec-1", repo, "brigade-implementer")


def test_execution_workspace_rejects_an_unregistered_directory(
    git_repo: tuple[Path, RouteState, ShadowWorktreeManager], tmp_path: Path,
):
    """A path that isn't a Git worktree the canonical checkout actually
    registered (e.g. some unrelated directory) must be rejected, not
    silently treated as an isolated workspace."""
    repo, state, _manager = git_repo
    unrelated = tmp_path / "not-a-worktree"
    unrelated.mkdir()
    _git(unrelated, "init", "-q")
    with pytest.raises(WorkspaceIsolationError, match="not a Git worktree registered"):
        _execution_workspace(state, "run-1", "epoch-1", "exec-1", unrelated, "brigade-implementer")


def test_execution_workspace_rejects_a_worktree_on_a_different_revision(
    git_repo: tuple[Path, RouteState, ShadowWorktreeManager], tmp_path: Path,
):
    """A registered worktree that has since moved to a different commit than
    the canonical checkout's current baseline must be rejected -- otherwise
    a stale or tampered worktree could pass as isolated."""
    repo, state, manager = git_repo
    worktree_path = tmp_path / "shadows" / "manual-worktree"
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    _git(repo, "worktree", "add", "--detach", str(worktree_path), "HEAD")

    # Advance the worktree to a new commit the canonical checkout doesn't have.
    (worktree_path / "extra.txt").write_text("drift\n", encoding="utf-8")
    _git(worktree_path, "add", "extra.txt")
    _git(worktree_path, "-c", "user.email=t@t.invalid", "-c", "user.name=t",
         "commit", "-qm", "drift")

    with pytest.raises(WorkspaceIsolationError, match="different canonical revision"):
        _execution_workspace(state, "run-1", "epoch-1", "exec-1", worktree_path, "brigade-implementer")


def test_execution_workspace_registers_a_valid_isolated_worktree(
    git_repo: tuple[Path, RouteState, ShadowWorktreeManager], tmp_path: Path,
):
    """The success path: a worktree that genuinely matches the canonical
    checkout's current baseline is accepted and registered as a shadow
    workspace owned by this execution."""
    repo, state, manager = git_repo
    baseline = manager.baseline()
    state.register_main_workspace(
        workspace_id=baseline.workspace_id, run_id="run-1", epoch_id="epoch-1",
        path=str(repo), base_sha=baseline.base_sha, dirty_patch_hash=baseline.dirty_patch_hash,
    )
    worktree_path = tmp_path / "shadows" / "valid-worktree"
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    _git(repo, "worktree", "add", "--detach", str(worktree_path), baseline.base_sha)

    workspace_id = _execution_workspace(
        state, "run-1", "epoch-1", "exec-1", worktree_path, "brigade-implementer",
    )
    assert workspace_id is not None
    workspace = state.get_workspace(workspace_id)
    assert workspace is not None
    assert workspace["kind"] == "shadow"
    assert workspace["status"] == "active"
    assert workspace["owner_execution_id"] == "exec-1"

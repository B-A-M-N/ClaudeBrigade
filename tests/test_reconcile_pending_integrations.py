"""Tests for ShadowWorktreeManager.reconcile_pending_integrations -- the
crash-recovery sweep for an integration journal left in 'applying' status
by a router process that died between applying a patch to the filesystem
and recording that success in the database. Previously untested.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from enhanced_router.shadow_worktree import ShadowWorktreeManager
from enhanced_router.state import RouteState


def _git(cwd: Path, *args: str, input_bytes: bytes | None = None) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, input=input_bytes,
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


def _register_main(state: RouteState, manager: ShadowWorktreeManager, repo: Path) -> dict:
    baseline = manager.baseline()
    return state.register_main_workspace(
        workspace_id=baseline.workspace_id, run_id="run-1", epoch_id="epoch-1",
        path=str(repo), base_sha=baseline.base_sha, dirty_patch_hash=baseline.dirty_patch_hash,
    )


def _capture_diff(repo: Path) -> bytes:
    """Like _git(), but returns raw bytes without .strip() -- a patch is
    whitespace-sensitive and _git()'s stripped text output corrupts it."""
    result = subprocess.run(
        ["git", "-C", str(repo), "diff"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    return result.stdout


def _seed_pending_candidate(state: RouteState, changeset_id: str) -> None:
    """The real flow always creates an integration_candidates row (via
    classify_overlap) before an integration attempt -- mark_integration_candidate
    is an UPDATE, so without a pre-existing row here it would silently affect
    zero rows."""
    state.create_integration_candidate(
        candidate_id=f"candidate-{changeset_id}", run_id="run-1", epoch_id="epoch-1",
        changeset_id=changeset_id, overlap={}, validation={"valid": True},
        disposition="green",
    )


def test_reconcile_recognizes_a_patch_that_was_actually_applied_before_the_crash(
    git_repo: tuple[Path, RouteState, ShadowWorktreeManager],
):
    """The patch reached the filesystem, but the process died before
    advance_canonical_workspace/finish_integration_journal recorded that in
    the DB. Reconciliation must recognize the already-applied patch via
    `git apply --reverse --check` and finish the journal as reconciled,
    without re-applying it or corrupting the working tree."""
    repo, state, manager = git_repo
    main = _register_main(state, manager, repo)
    pre_crash_dirty_hash = str(main["dirty_patch_hash"])

    # Build a real patch that changes tracked.txt, then actually apply it to
    # the working tree -- simulating the crash landing after the git apply
    # succeeded but before the router recorded success.
    (repo / "tracked.txt").write_text("integrated change\n", encoding="utf-8")
    patch_bytes = _capture_diff(repo)
    _git(repo, "checkout", "--", "tracked.txt")  # revert so the manual apply below is real
    apply_result = subprocess.run(
        ["git", "apply"], cwd=repo, input=patch_bytes,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    assert apply_result.returncode == 0, apply_result.stderr.decode()
    assert (repo / "tracked.txt").read_text(encoding="utf-8") == "integrated change\n"

    state.create_changeset(
        changeset_id="cs-1", execution_id="exec-1", workspace_id=str(main["workspace_id"]),
        base_sha=str(main["base_sha"]), patch_digest="digest-1", changed_files=["tracked.txt"],
        result={"validation": {"valid": True}}, status="validated", patch=patch_bytes,
    )
    _seed_pending_candidate(state, "cs-1")
    state.begin_integration_journal(
        journal_id="journal-1", run_id="run-1", epoch_id="epoch-1",
        workspace_id=str(main["workspace_id"]), changeset_id="cs-1",
        expected_generation=int(main.get("canonical_generation") or 0),
        expected_dirty_hash=pre_crash_dirty_hash,
    )
    # No finish_integration_journal call -- this is the "crashed mid-apply" state.

    result = ShadowWorktreeManager.reconcile_pending_integrations(state)
    assert result == {"reconciled": 1, "failed": 0}

    changeset = state.get_changeset("cs-1")
    assert changeset is not None
    assert changeset["status"] == "merged"
    candidates = state.get_integration_candidates(run_id="run-1", epoch_id="epoch-1")
    assert any(c["changeset_id"] == "cs-1" and c["disposition"] == "green" for c in candidates)
    # The working tree itself is untouched by reconciliation (it was already
    # applied pre-crash) -- still the integrated content, not re-applied or reverted.
    assert (repo / "tracked.txt").read_text(encoding="utf-8") == "integrated change\n"


def test_reconcile_marks_a_never_applied_patch_as_failed_not_silently_merged(
    git_repo: tuple[Path, RouteState, ShadowWorktreeManager],
):
    """The process crashed before git apply even ran -- the filesystem is
    still at the pre-integration baseline. Reconciliation must not guess
    this into a false success; it marks the journal failed and the
    candidate red so a human/controller sees it, rather than silently
    treating an un-integrated changeset as merged."""
    repo, state, manager = git_repo
    main = _register_main(state, manager, repo)
    pre_crash_dirty_hash = str(main["dirty_patch_hash"])

    # A real, valid patch that was simply never applied to the filesystem --
    # the crash happened before git apply ran.
    (repo / "tracked.txt").write_text("never applied\n", encoding="utf-8")
    patch_bytes = _capture_diff(repo)
    _git(repo, "checkout", "--", "tracked.txt")

    state.create_changeset(
        changeset_id="cs-2", execution_id="exec-2", workspace_id=str(main["workspace_id"]),
        base_sha=str(main["base_sha"]), patch_digest="digest-2", changed_files=["tracked.txt"],
        result={"validation": {"valid": True}}, status="validated", patch=patch_bytes,
    )
    _seed_pending_candidate(state, "cs-2")
    state.begin_integration_journal(
        journal_id="journal-2", run_id="run-1", epoch_id="epoch-1",
        workspace_id=str(main["workspace_id"]), changeset_id="cs-2",
        expected_generation=int(main.get("canonical_generation") or 0),
        expected_dirty_hash=pre_crash_dirty_hash,
    )

    result = ShadowWorktreeManager.reconcile_pending_integrations(state)
    assert result == {"reconciled": 0, "failed": 1}

    changeset = state.get_changeset("cs-2")
    assert changeset is not None
    assert changeset["status"] != "merged"
    candidates = state.get_integration_candidates(run_id="run-1", epoch_id="epoch-1")
    assert any(c["changeset_id"] == "cs-2" and c["disposition"] == "red" for c in candidates)
    assert (repo / "tracked.txt").read_text(encoding="utf-8") == "base\n"

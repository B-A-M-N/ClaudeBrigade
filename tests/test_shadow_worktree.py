"""Integration tests for the Git-backed shadow workspace boundary."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from enhanced_router.shadow_worktree import GitWorkspaceError, ShadowWorktreeManager
from enhanced_router.state import RouteState, WorkflowStateError


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


def test_shadow_preserves_dirty_baseline_without_stash_or_commit(
    git_repo: tuple[Path, RouteState, ShadowWorktreeManager],
) -> None:
    repo, state, manager = git_repo
    (repo / "tracked.txt").write_text("user change\n", encoding="utf-8")
    (repo / "user-notes.txt").write_text("keep this\n", encoding="utf-8")
    before_head = _git(repo, "rev-parse", "HEAD")
    before_status = _git(repo, "status", "--porcelain")
    baseline = manager.baseline()
    state.register_main_workspace(
        workspace_id=baseline.workspace_id,
        run_id="run-1",
        epoch_id="epoch-1",
        path=str(repo),
        base_sha=baseline.base_sha,
        dirty_patch_hash=baseline.dirty_patch_hash,
    )

    shadow = manager.create_shadow(
        state=state,
        run_id="run-1",
        epoch_id="epoch-1",
        execution_id="exec-1",
        allowed_dirty_patch_hash=baseline.dirty_patch_hash,
    )
    try:
        assert (shadow.path / "tracked.txt").read_text(encoding="utf-8") == "user change\n"
        assert (shadow.path / "user-notes.txt").read_text(encoding="utf-8") == "keep this\n"
        assert _git(repo, "rev-parse", "HEAD") == before_head
        assert _git(repo, "status", "--porcelain") == before_status
        assert state.get_workspace(shadow.workspace_id)["kind"] == "shadow"  # type: ignore[index]
    finally:
        manager.remove_shadow(shadow.path)
        state.update_workspace_status(shadow.workspace_id, "discarded")


def test_changeset_is_validated_and_green_changes_apply_to_main(
    git_repo: tuple[Path, RouteState, ShadowWorktreeManager],
) -> None:
    repo, state, manager = git_repo
    baseline = manager.baseline()
    state.register_main_workspace(
        workspace_id=baseline.workspace_id,
        run_id="run-1",
        epoch_id="epoch-1",
        path=str(repo),
        base_sha=baseline.base_sha,
        dirty_patch_hash=baseline.dirty_patch_hash,
    )
    shadow = manager.create_shadow(
        state=state, run_id="run-1", epoch_id="epoch-1", execution_id="exec-1",
        allowed_dirty_patch_hash=baseline.dirty_patch_hash,
    )
    try:
        (shadow.path / "tracked.txt").write_text("base\nworker change\n", encoding="utf-8")
        changeset = manager.extract_changeset(state=state, workspace=shadow)
        assert changeset.validation["valid"] is True
        assert changeset.changed_files == ("tracked.txt",)
        stored = state.get_changeset(changeset.changeset_id)
        assert stored is not None
        assert stored["patch_blob"] == changeset.patch
        classification = manager.classify_overlap(
            state=state, run_id="run-1", epoch_id="epoch-1", changeset=changeset,
        )
        assert classification["disposition"] == "green"
        state.mark_integration_candidate(changeset.changeset_id, disposition="yellow")
        actions = state.get_runnable_actions("run-1", "epoch-1")
        assert any(
            item.get("action_kind") == "controller_integration"
            and item.get("changeset_id") == changeset.changeset_id
            for item in actions
        )
        integration_action = next(
            item for item in actions
            if item.get("changeset_id") == changeset.changeset_id
        )
        claimed = state.claim_runnable_action(
            "run-1", "epoch-1", str(integration_action["action_id"]),
        )
        assert claimed["requires_main_controller"] is True
        assert state.consume_controller_action(
            "run-1", "epoch-1", str(integration_action["action_id"]),
        ) is not None
        result = manager.integrate_green(
            state=state, run_id="run-1", epoch_id="epoch-1", changeset=changeset,
            expected_dirty_patch_hash=baseline.dirty_patch_hash,
        )
        assert result["status"] == "applied"
        assert (repo / "tracked.txt").read_text(encoding="utf-8") == "base\nworker change\n"
        assert state.get_changeset(changeset.changeset_id)["status"] == "merged"  # type: ignore[index]
    finally:
        manager.remove_shadow(shadow.path)


def test_green_changeset_preserves_existing_main_dirty_edits(
    git_repo: tuple[Path, RouteState, ShadowWorktreeManager],
) -> None:
    repo, state, manager = git_repo
    (repo / "tracked.txt").write_text("user baseline\n", encoding="utf-8")
    (repo / "notes.txt").write_text("keep\n", encoding="utf-8")
    baseline = manager.baseline()
    state.register_main_workspace(
        workspace_id=baseline.workspace_id, run_id="run-1", epoch_id="epoch-1",
        path=str(repo), base_sha=baseline.base_sha,
        dirty_patch_hash=baseline.dirty_patch_hash,
        baseline_untracked_files=list(baseline.untracked_files),
    )
    shadow = manager.create_shadow(
        state=state, run_id="run-1", epoch_id="epoch-1", execution_id="exec-dirty",
        allowed_dirty_patch_hash=baseline.dirty_patch_hash,
    )
    try:
        (shadow.path / "tracked.txt").write_text("user baseline\nworker delta\n", encoding="utf-8")
        changeset = manager.extract_changeset(state=state, workspace=shadow)
        assert changeset.validation["valid"] is True
        manager.integrate_green(
            state=state, run_id="run-1", epoch_id="epoch-1", changeset=changeset,
            expected_dirty_patch_hash=baseline.dirty_patch_hash,
        )
        assert (repo / "tracked.txt").read_text(encoding="utf-8") == "user baseline\nworker delta\n"
        assert (repo / "notes.txt").read_text(encoding="utf-8") == "keep\n"
    finally:
        manager.remove_shadow(shadow.path)


def test_changeset_includes_worker_created_untracked_file(
    git_repo: tuple[Path, RouteState, ShadowWorktreeManager],
) -> None:
    repo, state, manager = git_repo
    baseline = manager.baseline()
    state.register_main_workspace(
        workspace_id=baseline.workspace_id, run_id="run-1", epoch_id="epoch-1",
        path=str(repo), base_sha=baseline.base_sha, dirty_patch_hash=baseline.dirty_patch_hash,
    )
    shadow = manager.create_shadow(
        state=state, run_id="run-1", epoch_id="epoch-1", execution_id="exec-new",
    )
    try:
        (shadow.path / "new.py").write_text("answer = 42\n", encoding="utf-8")
        changeset = manager.extract_changeset(state=state, workspace=shadow)
        assert changeset.validation["valid"] is True
        assert "new.py" in changeset.changed_files
        manager.integrate_green(
            state=state, run_id="run-1", epoch_id="epoch-1", changeset=changeset,
            expected_dirty_patch_hash=baseline.dirty_patch_hash,
        )
        assert (repo / "new.py").read_text(encoding="utf-8") == "answer = 42\n"
    finally:
        manager.remove_shadow(shadow.path)


def test_cross_run_main_workspace_collision_is_rejected(
    git_repo: tuple[Path, RouteState, ShadowWorktreeManager],
) -> None:
    repo, state, manager = git_repo
    state.create_run("run-2", session_id="session-2", cwd=str(repo))
    baseline = manager.baseline()
    state.register_main_workspace(
        workspace_id=baseline.workspace_id, run_id="run-1", epoch_id="epoch-1",
        path=str(repo), base_sha=baseline.base_sha, dirty_patch_hash=baseline.dirty_patch_hash,
    )
    with pytest.raises(WorkflowStateError):
        state.register_main_workspace(
            workspace_id=baseline.workspace_id, run_id="run-2", epoch_id="epoch-2",
            path=str(repo), base_sha=baseline.base_sha, dirty_patch_hash=baseline.dirty_patch_hash,
        )


def test_changed_main_baseline_blocks_integration(
    git_repo: tuple[Path, RouteState, ShadowWorktreeManager],
) -> None:
    repo, state, manager = git_repo
    baseline = manager.baseline()
    state.register_main_workspace(
        workspace_id=baseline.workspace_id, run_id="run-1", epoch_id="epoch-1",
        path=str(repo), base_sha=baseline.base_sha, dirty_patch_hash=baseline.dirty_patch_hash,
    )
    shadow = manager.create_shadow(
        state=state, run_id="run-1", epoch_id="epoch-1", execution_id="exec-1",
    )
    try:
        (shadow.path / "tracked.txt").write_text("base\nworker\n", encoding="utf-8")
        changeset = manager.extract_changeset(state=state, workspace=shadow)
        (repo / "tracked.txt").write_text("main changed\n", encoding="utf-8")
        with pytest.raises(GitWorkspaceError, match="dirty state changed"):
            manager.integrate_green(
                state=state, run_id="run-1", epoch_id="epoch-1", changeset=changeset,
                expected_dirty_patch_hash=baseline.dirty_patch_hash,
            )
    finally:
        manager.remove_shadow(shadow.path)


def test_sequential_green_integrations_advance_canonical_generation(
    git_repo: tuple[Path, RouteState, ShadowWorktreeManager],
) -> None:
    repo, state, manager = git_repo
    baseline = manager.baseline()
    state.register_main_workspace(
        workspace_id=baseline.workspace_id, run_id="run-1", epoch_id="epoch-1",
        path=str(repo), base_sha=baseline.base_sha,
        dirty_patch_hash=baseline.dirty_patch_hash,
    )

    first = manager.create_shadow(
        state=state, run_id="run-1", epoch_id="epoch-1", execution_id="exec-first",
        allowed_dirty_patch_hash=baseline.dirty_patch_hash,
    )
    try:
        (first.path / "tracked.txt").write_text("first\n", encoding="utf-8")
        first_changeset = manager.extract_changeset(state=state, workspace=first)
        manager.integrate_green(
            state=state, run_id="run-1", epoch_id="epoch-1",
            changeset=first_changeset,
            expected_dirty_patch_hash=baseline.dirty_patch_hash,
        )
    finally:
        manager.remove_shadow(first.path)
        state.update_workspace_status(first.workspace_id, "merged")

    current = manager.baseline()
    second = manager.create_shadow(
        state=state, run_id="run-1", epoch_id="epoch-1", execution_id="exec-second",
        allowed_dirty_patch_hash=current.dirty_patch_hash,
    )
    try:
        (second.path / "tracked.txt").write_text("first\nsecond\n", encoding="utf-8")
        second_changeset = manager.extract_changeset(state=state, workspace=second)
        result = manager.integrate_green(
            state=state, run_id="run-1", epoch_id="epoch-1",
            changeset=second_changeset,
            expected_dirty_patch_hash=current.dirty_patch_hash,
        )
        assert result["status"] == "applied"
        main = state.get_workspace(baseline.workspace_id)
        assert main is not None
        assert main["canonical_generation"] == 2
        assert (repo / "tracked.txt").read_text(encoding="utf-8") == "first\nsecond\n"
    finally:
        manager.remove_shadow(second.path)
        state.update_workspace_status(second.workspace_id, "merged")

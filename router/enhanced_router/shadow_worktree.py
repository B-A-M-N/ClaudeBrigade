"""Git-backed shadow workspaces and deterministic changeset integration.

The Claude Code ``Agent`` tool is the process that owns a worker lifecycle;
this module does not try to spawn or replay agents.  It owns the filesystem
boundary around a mutating execution:

* capture the user's current ``HEAD`` and dirty state without stashing or
  committing it;
* create a detached shadow worktree from that exact baseline;
* copy the baseline's untracked files into the shadow;
* extract a bounded, auditable changeset from the worker workspace; and
* apply a validated changeset to the original workspace only when its
  baseline and dirty patch still match.

Integration deliberately leaves the caller's normal Git commit workflow
intact.  It applies a patch, but never creates a commit, stash, reset, merge,
or branch in the user's main worktree.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


class GitWorkspaceError(RuntimeError):
    """Raised when a workspace operation cannot be completed safely."""


@dataclass(frozen=True)
class GitResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass(frozen=True)
class WorkspaceBaseline:
    root: Path
    common_dir: Path
    workspace_id: str
    base_sha: str
    dirty_patch_hash: str
    dirty_patch: bytes
    untracked_files: tuple[str, ...]


@dataclass(frozen=True)
class ShadowWorkspace:
    workspace_id: str
    path: Path
    base_sha: str
    dirty_patch_hash: str
    run_id: str
    epoch_id: str
    execution_id: str
    baseline_untracked_files: tuple[str, ...] = ()
    parent_canonical_generation: int | None = None


@dataclass(frozen=True)
class Changeset:
    changeset_id: str
    execution_id: str
    workspace_id: str
    base_sha: str
    patch_digest: str
    patch: bytes
    changed_files: tuple[str, ...]
    validation: dict[str, object]
    parent_canonical_generation: int | None = None


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_relative(root: Path, value: str) -> Path:
    """Resolve a Git path and reject path traversal."""
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise GitWorkspaceError(f"Git path escapes repository: {value!r}") from exc
    return path


class ShadowWorktreeManager:
    """Manage shadow worktrees for one canonical Git repository."""

    def __init__(self, repo_path: str | Path, *, worktree_root: str | Path | None = None) -> None:
        self.repo_path = self.repository_root(Path(repo_path))
        self.worktree_root = Path(worktree_root) if worktree_root else (
            self.repo_path / ".claude-brigade" / "worktrees"
        )

    @staticmethod
    def _run_git(
        cwd: Path,
        *args: str,
        input_bytes: bytes | None = None,
        timeout: float = 30.0,
    ) -> GitResult:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=str(cwd),
                input=input_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GitWorkspaceError(f"git {' '.join(args)} failed: {exc}") from exc
        return GitResult(
            args=tuple(args),
            returncode=completed.returncode,
            stdout=completed.stdout.decode("utf-8", "replace"),
            stderr=completed.stderr.decode("utf-8", "replace"),
        )

    @classmethod
    def repository_root(cls, path: Path) -> Path:
        result = cls._run_git(path, "rev-parse", "--show-toplevel")
        if not result.ok:
            raise GitWorkspaceError(result.stderr.strip() or "not a Git worktree")
        root = Path(result.stdout.strip()).resolve()
        if not root.is_dir():
            raise GitWorkspaceError(f"Git root is not a directory: {root}")
        return root

    def is_registered_worktree(self, path: Path) -> bool:
        """Return whether *path* is a worktree of this canonical checkout."""
        target = path.resolve()
        result = self._run_git(self.repo_path, "worktree", "list", "--porcelain")
        if not result.ok:
            raise GitWorkspaceError(
                result.stderr.strip() or "unable to enumerate Git worktrees"
            )
        roots = {
            Path(line.removeprefix("worktree ")).resolve()
            for line in result.stdout.splitlines()
            if line.startswith("worktree ")
        }
        return target in roots

    def _require_clean_worktree_path(self, path: Path) -> None:
        if path.exists():
            if any(path.iterdir()):
                raise GitWorkspaceError(f"shadow worktree path is not empty: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)

    def _tracked_patch(self) -> bytes:
        result = self._run_git(self.repo_path, "diff", "--binary", "HEAD")
        if not result.ok:
            raise GitWorkspaceError(result.stderr.strip() or "unable to read Git diff")
        return result.stdout.encode("utf-8")

    def _untracked_files(self) -> tuple[str, ...]:
        result = self._run_git(
            self.repo_path, "ls-files", "--others", "--exclude-standard", "-z"
        )
        if not result.ok:
            raise GitWorkspaceError(result.stderr.strip() or "unable to list untracked files")
        values = tuple(item for item in result.stdout.split("\0") if item)
        for value in values:
            _safe_relative(self.repo_path, value)
        return values

    def baseline(self) -> WorkspaceBaseline:
        base = self._run_git(self.repo_path, "rev-parse", "HEAD")
        common = self._run_git(self.repo_path, "rev-parse", "--git-common-dir")
        if not base.ok or not common.ok:
            raise GitWorkspaceError(base.stderr.strip() or common.stderr.strip() or "Git baseline unavailable")
        base_sha = base.stdout.strip()
        common_dir = Path(common.stdout.strip())
        if not common_dir.is_absolute():
            common_dir = (self.repo_path / common_dir).resolve()
        else:
            common_dir = common_dir.resolve()
        untracked = self._untracked_files()
        tracked_patch = self._tracked_patch()
        digest_input = bytearray(tracked_patch)
        digest_input.extend(b"\0--untracked--\0")
        for relative in untracked:
            path = _safe_relative(self.repo_path, relative)
            digest_input.extend(relative.encode("utf-8"))
            digest_input.extend(b"\0")
            try:
                digest_input.extend(path.read_bytes())
            except OSError as exc:
                raise GitWorkspaceError(f"cannot read untracked baseline file {relative!r}: {exc}") from exc
            digest_input.extend(b"\0")
        workspace_id = "git-" + _digest_bytes(str(common_dir).encode("utf-8"))[:24]
        return WorkspaceBaseline(
            root=self.repo_path,
            common_dir=common_dir,
            workspace_id=workspace_id,
            base_sha=base_sha,
            dirty_patch_hash=_digest_bytes(bytes(digest_input)),
            dirty_patch=tracked_patch,
            untracked_files=untracked,
        )

    def _copy_untracked(self, baseline: WorkspaceBaseline, destination: Path) -> None:
        for relative in baseline.untracked_files:
            source = _safe_relative(self.repo_path, relative)
            target = _safe_relative(destination, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    def _path_patch(self, before: Path | None, after: Path, relative: str) -> bytes:
        """Return a patch from the canonical baseline to a worker path.

        Comparing the two worktrees directly is important: the worker starts
        with the user's dirty baseline applied on top of ``HEAD``.  A patch
        generated against ``HEAD`` would contain the user's pre-existing
        edits and would be unsafe to apply back to the already-dirty main
        worktree.
        """
        left = str(before) if before is not None and before.exists() else os.devnull
        right = str(after) if after.exists() else os.devnull
        result = self._run_git(
            after.parent,
            "diff", "--no-index", "--binary", left, right,
        )
        # ``git diff --no-index`` returns 1 for a difference, which is the
        # expected result.  Any other non-zero result is an operational error.
        if result.returncode not in {0, 1}:
            raise GitWorkspaceError(result.stderr.strip() or "unable to diff untracked file")
        if result.returncode == 0:
            return b""
        lines = result.stdout.splitlines(keepends=True)
        normalized: list[str] = []
        for line in lines:
            if line.startswith("diff --git "):
                normalized.append(f"diff --git a/{relative} b/{relative}\n")
            elif line.startswith("--- "):
                normalized.append(
                    "--- /dev/null\n" if before is None or not before.exists()
                    else f"--- a/{relative}\n"
                )
            elif line.startswith("+++ "):
                normalized.append(f"+++ b/{relative}\n" if after.exists() else "+++ /dev/null\n")
            else:
                normalized.append(line)
        return "".join(normalized).encode("utf-8")

    def create_shadow(
        self,
        *,
        state: object,
        run_id: str,
        epoch_id: str,
        execution_id: str,
        allowed_dirty_patch_hash: str | None = None,
    ) -> ShadowWorkspace:
        """Create and register a detached shadow worktree.

        The main worktree is never stashed or committed.  A caller may pass
        ``allowed_dirty_patch_hash`` when it has already registered the main
        workspace; a changed dirty baseline is rejected rather than silently
        copied into a worker.
        """
        baseline = self.baseline()
        if allowed_dirty_patch_hash is not None and baseline.dirty_patch_hash != allowed_dirty_patch_hash:
            raise GitWorkspaceError("main worktree changed after its registered baseline")
        workspace_id = f"shadow-{uuid.uuid4().hex}"
        path = self.worktree_root / run_id / execution_id
        self._require_clean_worktree_path(path)
        added = self._run_git(self.repo_path, "worktree", "add", "--detach", str(path), baseline.base_sha)
        if not added.ok:
            raise GitWorkspaceError(added.stderr.strip() or "unable to create shadow worktree")
        try:
            if baseline.dirty_patch:
                applied = self._run_git(path, "apply", "--binary", input_bytes=baseline.dirty_patch)
                if not applied.ok:
                    raise GitWorkspaceError(applied.stderr.strip() or "unable to apply tracked baseline patch")
            self._copy_untracked(baseline, path)
            main_rows = state.get_workspaces(  # type: ignore[attr-defined]
                run_id=run_id, epoch_id=epoch_id, kind="main", status="active",
            )
            main = main_rows[0] if main_rows else None
            state.create_workspace(  # type: ignore[attr-defined]
                workspace_id=workspace_id,
                run_id=run_id,
                epoch_id=epoch_id,
                kind="shadow",
                path=str(path),
                base_sha=baseline.base_sha,
                dirty_patch_hash=baseline.dirty_patch_hash,
                status="active",
                owner_execution_id=execution_id,
                baseline_untracked_files=list(baseline.untracked_files),
                parent_canonical_generation=(
                    int(main.get("canonical_generation") or 0) if main else None
                ),
                parent_dirty_patch_hash=(
                    str(main.get("current_dirty_hash") or main.get("dirty_patch_hash") or "")
                    if main else None
                ),
            )
        except Exception:
            self.remove_shadow(path)
            raise
        return ShadowWorkspace(
            workspace_id=workspace_id,
            path=path,
            base_sha=baseline.base_sha,
            dirty_patch_hash=baseline.dirty_patch_hash,
            run_id=run_id,
            epoch_id=epoch_id,
            execution_id=execution_id,
            baseline_untracked_files=baseline.untracked_files,
            parent_canonical_generation=(
                int(main.get("canonical_generation") or 0) if main else None
            ),
        )

    def register_main(self, *, state: object, run_id: str, epoch_id: str) -> dict:
        """Register the canonical worktree and reject cross-run collisions."""
        baseline = self.baseline()
        return state.register_main_workspace(  # type: ignore[attr-defined]
            workspace_id=baseline.workspace_id,
            run_id=run_id,
            epoch_id=epoch_id,
            path=str(self.repo_path),
            base_sha=baseline.base_sha,
            dirty_patch_hash=baseline.dirty_patch_hash,
            baseline_untracked_files=list(baseline.untracked_files),
        )

    def _changed_files(self, base_sha: str) -> tuple[str, ...]:
        result = self._run_git(self.repo_path, "diff", "--name-only", "-z", base_sha)
        if not result.ok:
            raise GitWorkspaceError(result.stderr.strip() or "unable to list changed files")
        values = tuple(item for item in result.stdout.split("\0") if item)
        return values

    def extract_changeset(
        self,
        *,
        state: object,
        workspace: ShadowWorkspace,
        changeset_id: str | None = None,
        allowed_files: Iterable[str] | None = None,
    ) -> Changeset:
        """Capture a worker patch without requiring a worker commit."""
        changed = self._run_git(workspace.path, "diff", "--name-only", "-z", workspace.base_sha)
        if not changed.ok:
            raise GitWorkspaceError(changed.stderr.strip() or "unable to list changeset files")
        changed_files: list[str] = []
        tracked_candidates = [item for item in changed.stdout.split("\0") if item]
        for relative in tracked_candidates:
            before = _safe_relative(self.repo_path, relative)
            after = _safe_relative(workspace.path, relative)
            if self._path_patch(before, after, relative):
                changed_files.append(relative)
        baseline_untracked = set(workspace.baseline_untracked_files)
        current_untracked = self._run_git(
            workspace.path, "ls-files", "--others", "--exclude-standard", "-z"
        )
        if not current_untracked.ok:
            raise GitWorkspaceError(current_untracked.stderr.strip() or "unable to list worker untracked files")
        current_untracked_files = {
            item for item in current_untracked.stdout.split("\0") if item
        }
        for relative in sorted(baseline_untracked | current_untracked_files):
            before = _safe_relative(self.repo_path, relative) if relative in baseline_untracked else None
            after = _safe_relative(workspace.path, relative)
            before_digest = _digest_bytes(before.read_bytes()) if before is not None and before.exists() else None
            after_digest = _digest_bytes(after.read_bytes()) if after.exists() else None
            if before_digest == after_digest:
                continue
            changed_files.append(relative)
        changed_files = list(sorted(set(changed_files)))
        patch_parts: list[bytes] = []
        for relative in changed_files:
            before = _safe_relative(self.repo_path, relative)
            after = _safe_relative(workspace.path, relative)
            path_patch = self._path_patch(before, after, relative)
            if path_patch:
                patch_parts.append(path_patch)
        patch = b"".join(patch_parts)
        allowed = {str(Path(item)) for item in allowed_files} if allowed_files is not None else None
        unauthorized = [item for item in changed_files if allowed is not None and item not in allowed]
        patch_check = self._run_git(
            self.repo_path, "apply", "--check", "--whitespace=error", input_bytes=patch,
        ) if patch else GitResult((), 0, "", "")
        validation = {
            "patch_nonempty": bool(patch),
            "diff_check": patch_check.ok,
            "unauthorized_files": unauthorized,
            "changed_files": list(changed_files),
        }
        if unauthorized:
            validation["valid"] = False
            validation["reason"] = "changeset modifies files outside its ownership scope"
        elif not patch_check.ok:
            validation["valid"] = False
            validation["reason"] = patch_check.stderr.strip() or "changeset patch preflight failed"
        else:
            validation["valid"] = True
        changeset_id = changeset_id or f"changeset-{uuid.uuid4().hex}"
        patch_digest = _digest_bytes(patch)
        state.create_changeset(  # type: ignore[attr-defined]
            changeset_id=changeset_id,
            execution_id=workspace.execution_id,
            workspace_id=workspace.workspace_id,
            base_sha=workspace.base_sha,
            patch_digest=patch_digest,
            changed_files=list(changed_files),
            result={"validation": validation},
            status="validated" if validation["valid"] else "rejected",
            patch=patch,
            parent_canonical_generation=(
                int(getattr(workspace, "parent_canonical_generation", 0) or 0)
            ),
        )
        return Changeset(
            changeset_id=changeset_id,
            execution_id=workspace.execution_id,
            workspace_id=workspace.workspace_id,
            base_sha=workspace.base_sha,
            patch_digest=patch_digest,
            patch=patch,
            changed_files=tuple(changed_files),
            validation=validation,
            parent_canonical_generation=(
                int(getattr(workspace, "parent_canonical_generation", 0) or 0)
            ),
        )

    def classify_overlap(self, *, state: object, run_id: str, epoch_id: str, changeset: Changeset) -> dict:
        """Classify path overlap with earlier changesets deterministically."""
        prior = state.get_changesets(run_id=run_id, epoch_id=epoch_id)  # type: ignore[attr-defined]
        current = set(changeset.changed_files)
        overlaps: dict[str, list[str]] = {}
        for item in prior:
            if item.get("changeset_id") == changeset.changeset_id or item.get("status") in {"rejected"}:
                continue
            files = set(json.loads(item.get("changed_files_json") or "[]"))
            common = sorted(current & files)
            if common:
                overlaps[str(item["changeset_id"])] = common
        if not changeset.validation.get("valid", False):
            disposition = "red"
        elif overlaps:
            disposition = "yellow"
        else:
            disposition = "green"
        payload = {"overlaps": overlaps, "changed_files": sorted(current)}
        validation = dict(changeset.validation)
        validation["overlap_free"] = not overlaps
        state.create_integration_candidate(  # type: ignore[attr-defined]
            candidate_id=f"candidate-{uuid.uuid4().hex}",
            run_id=run_id,
            epoch_id=epoch_id,
            changeset_id=changeset.changeset_id,
            overlap=payload,
            validation=validation,
            disposition=disposition,
        )
        return {"disposition": disposition, "overlap": payload, "validation": validation}

    def integrate_green(
        self,
        *,
        state: object,
        run_id: str,
        epoch_id: str,
        changeset: Changeset,
        expected_dirty_patch_hash: str,
    ) -> dict:
        """Apply a green changeset to the original worktree, without commit."""
        if not changeset.validation.get("valid", False):
            raise GitWorkspaceError("cannot integrate an invalid changeset")
        if not changeset.patch:
            state.mark_changeset_merged(changeset.changeset_id)  # type: ignore[attr-defined]
            state.mark_integration_candidate(  # type: ignore[attr-defined]
                changeset.changeset_id,
                disposition="green",
                validation={"applied": False, "no_changes": True},
            )
            return {
                "changeset_id": changeset.changeset_id,
                "base_sha": changeset.base_sha,
                "changed_files": [],
                "status": "no_changes",
            }
        baseline = self.baseline()
        main_rows = state.get_workspaces(  # type: ignore[attr-defined]
            run_id=run_id, epoch_id=epoch_id, kind="main", status="active",
        )
        main = main_rows[0] if main_rows else None
        if main is None:
            raise GitWorkspaceError("canonical workspace is not registered")
        expected_generation = int(main.get("canonical_generation") or 0)
        parent_generation = getattr(changeset, "parent_canonical_generation", None)
        if parent_generation is not None and int(parent_generation) != expected_generation:
            raise GitWorkspaceError("changeset is based on a stale canonical generation")
        current_hash = str(main.get("current_dirty_hash") or main.get("dirty_patch_hash") or "")
        if baseline.dirty_patch_hash != expected_dirty_patch_hash or current_hash != expected_dirty_patch_hash:
            raise GitWorkspaceError("main worktree dirty state changed before integration")
        if baseline.base_sha != changeset.base_sha:
            raise GitWorkspaceError(
                f"changeset baseline {changeset.base_sha} does not match main HEAD {baseline.base_sha}"
            )
        status = self._run_git(self.repo_path, "status", "--porcelain")
        # The user's baseline is expected to be present.  It is safe to apply
        # a worker delta only when it is exactly the state captured at intake.
        if not status.ok:
            raise GitWorkspaceError(status.stderr.strip() or "unable to inspect main worktree")
        preflight = self._run_git(
            self.repo_path, "apply", "--binary", "--check", "--whitespace=error",
            input_bytes=changeset.patch,
        )
        if not preflight.ok:
            raise GitWorkspaceError(
                preflight.stderr.strip() or "changeset failed the main-worktree conflict check"
            )
        journal_id = f"integration:{changeset.changeset_id}"
        state.begin_integration_journal(  # type: ignore[attr-defined]
            journal_id=journal_id, run_id=run_id, epoch_id=epoch_id,
            workspace_id=str(main["workspace_id"]) if main else "",
            changeset_id=changeset.changeset_id,
            expected_generation=expected_generation,
            expected_dirty_hash=expected_dirty_patch_hash,
        )
        # Once the journal is 'applying', begin_integration_journal refuses any
        # other integration against this workspace, so every path below --
        # including an unexpected exception -- must terminalize it.  Otherwise
        # the workspace stays locked out of integration until process restart.
        try:
            applied = self._run_git(self.repo_path, "apply", "--binary", input_bytes=changeset.patch)
            if not applied.ok:
                raise GitWorkspaceError(applied.stderr.strip() or "changeset could not be applied cleanly")
            after = self.baseline()
            if main is not None:
                state.advance_canonical_workspace(  # type: ignore[attr-defined]
                    workspace_id=str(main["workspace_id"]),
                    expected_generation=expected_generation,
                    expected_dirty_hash=expected_dirty_patch_hash,
                    applied_changeset_id=changeset.changeset_id,
                    new_base_sha=after.base_sha,
                    new_dirty_hash=after.dirty_patch_hash,
                )
            state.mark_changeset_merged(changeset.changeset_id)  # type: ignore[attr-defined]
            state.mark_integration_candidate(  # type: ignore[attr-defined]
                changeset.changeset_id, disposition="green", validation={"applied": True}
            )
            state.finish_integration_journal(journal_id, "completed")  # type: ignore[attr-defined]
        except Exception as exc:
            state.finish_integration_journal(journal_id, "failed", str(exc))  # type: ignore[attr-defined]
            raise
        return {
            "changeset_id": changeset.changeset_id,
            "base_sha": changeset.base_sha,
            "changed_files": list(changeset.changed_files),
            "status": "applied",
        }

    @classmethod
    def reconcile_pending_integrations(cls, state: object) -> dict[str, int]:
        """Reconcile filesystem effects left by a crash during integration.

        A journal is only marked reconciled when the persisted patch is
        demonstrably present in the canonical checkout (``git apply
        --reverse --check``).  Ambiguous state is surfaced as a failed/red
        candidate and is never guessed into the database as successfully
        merged.
        """
        reconciled = 0
        failed = 0
        for journal in state.get_pending_integration_journals():  # type: ignore[attr-defined]
            journal_id = str(journal["journal_id"])
            changeset = state.get_changeset(str(journal["changeset_id"]))  # type: ignore[attr-defined]
            main = state.get_workspace(str(journal["workspace_id"]))  # type: ignore[attr-defined]
            if changeset is None or main is None:
                state.finish_integration_journal(  # type: ignore[attr-defined]
                    journal_id, "failed", "integration references missing state",
                )  # type: ignore[attr-defined]
                failed += 1
                continue
            patch_blob = changeset.get("patch_blob")
            if not isinstance(patch_blob, (bytes, bytearray)) or not patch_blob:
                state.finish_integration_journal(  # type: ignore[attr-defined]
                    journal_id, "failed", "integration patch is missing or empty",
                )  # type: ignore[attr-defined]
                failed += 1
                continue
            manager = cls(str(main["path"]))
            current = manager.baseline()
            expected_hash = str(journal.get("expected_dirty_hash") or "")
            if current.dirty_patch_hash == expected_hash:
                state.finish_integration_journal(  # type: ignore[attr-defined]
                    journal_id, "failed", "filesystem was unchanged when integration journal was recovered",
                )  # type: ignore[attr-defined]
                state.mark_integration_candidate(  # type: ignore[attr-defined]
                    str(changeset["changeset_id"]), disposition="red",
                    validation={"recovery": "patch not applied"},
                )
                failed += 1
                continue
            reverse = manager._run_git(
                manager.repo_path, "apply", "--binary", "--reverse", "--check",
                input_bytes=bytes(patch_blob),
            )
            if not reverse.ok:
                state.finish_integration_journal(  # type: ignore[attr-defined]
                    journal_id, "failed", "filesystem differs but patch presence is ambiguous",
                )  # type: ignore[attr-defined]
                state.mark_integration_candidate(  # type: ignore[attr-defined]
                    str(changeset["changeset_id"]), disposition="red",
                    validation={"recovery": "ambiguous filesystem state"},
                )
                failed += 1
                continue
            try:
                state.advance_canonical_workspace(  # type: ignore[attr-defined]
                    workspace_id=str(main["workspace_id"]),
                    expected_generation=int(journal["expected_generation"]),
                    expected_dirty_hash=expected_hash,
                    applied_changeset_id=str(changeset["changeset_id"]),
                    new_base_sha=current.base_sha,
                    new_dirty_hash=current.dirty_patch_hash,
                )
                state.mark_changeset_merged(str(changeset["changeset_id"]))  # type: ignore[attr-defined]
                state.mark_integration_candidate(  # type: ignore[attr-defined]
                    str(changeset["changeset_id"]), disposition="green",
                    validation={"recovery": "patch presence verified"},
                )
                state.finish_integration_journal(journal_id, "reconciled")  # type: ignore[attr-defined]
                reconciled += 1
            except Exception as exc:
                state.finish_integration_journal(journal_id, "failed", str(exc))  # type: ignore[attr-defined]
                failed += 1
        return {"reconciled": reconciled, "failed": failed}

    def remove_shadow(self, path: str | Path) -> None:
        shadow = Path(path)
        if not shadow.exists():
            return
        result = self._run_git(self.repo_path, "worktree", "remove", "--force", str(shadow))
        if not result.ok:
            raise GitWorkspaceError(result.stderr.strip() or f"unable to remove shadow worktree {shadow}")

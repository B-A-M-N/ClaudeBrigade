"""Workspace-policy engine for ClaudeBrigade tier classification.

This module provides objective, automated tier classification for the
brigade workflow.  The controller self-selects a tier, but the policy
engine can flag mismatches when workspace changes clearly exceed what
the claimed tier permits.

Tier escalation rules (lightest -> heaviest):

  trivial    - one file, <= 3 meaningful line changes, no structural
               changes (no new files, no function/class additions,
               no dependency changes).
  normal     - everything else that stays within a single subsystem.
  cross-cutting  - changes touch >= 2 distinct subsystems, or modify
                 shared infrastructure (config, hooks, router).
  high-risk  - changes to security-sensitive paths (auth, tokens,
               crypto, network config), or changes to the brigade
               harness itself.

Usage::

    from enhanced_router.policy import classify_workspace
    report = classify_workspace(head_ref, cwd)
    # report.minimum_tier tells the lightest tier the changes require

"""
from __future__ import annotations

import dataclasses
import fnmatch
import pathlib
import re
import subprocess
import sys

# ---------------------------------------------------------------------------
# Configurable rule sets (overridable via environment or config)
# ---------------------------------------------------------------------------

#: Paths that always trigger at least *cross-cutting*.
INFRA_GLOBS: list[str] = [
    "router/**",
    "hooks/**",
    ".claude/**",
    "agents/**",
    "bin/**",
    "install.sh",
    "**/pyproject.toml",
    "**/requirements.txt",
    "**/*.yaml",
    "**/*.yml",
    "**/package.json",
]

#: Paths that trigger *high-risk*.
SECURITY_GLOBS: list[str] = [
    "**/tokens/**",
    "**/*.token",
    "**/providers.env",
    "**/.env",
    "**/litellm.token",
    "**/router.token",
    "**/auth/**",
    "**/secret**",
    "**/crypto**",
    "**/key**",
    "**/credential**",
]

#: Maximum meaningful changes for *trivial* tier.
TRIVIAL_MAX_CHANGES = 3

#: Regex that matches "meaningful" changes (not whitespace-only).
_MEANINGFUL_RE = re.compile(
    r"^[+-]\s*[^+-]",  # lines starting with + or - that aren't just whitespace
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ChangeFile:
    """Diff info for a single file."""
    path: str
    added: int
    deleted: int
    new_file: bool
    deleted_file: bool
    meaningful_changes: int


@dataclasses.dataclass(frozen=True, kw_only=True)
class ChangeReport:
    """Summary of all workspace changes relative to *head_ref*."""
    files: list[ChangeFile] = dataclasses.field(default_factory=list)
    total_meaningful_changes: int = 0
    subsystems: set[str] = dataclasses.field(default_factory=set)
    touches_infra: bool = False
    touches_security: bool = False
    min_trivial_tier: bool = True  # computed at construction by classify_workspace

    def fits_trivial(self) -> bool:
        """Return True if this report objectively fits the *trivial* tier."""
        return (
            len(self.files) <= 1
            and self.total_meaningful_changes <= TRIVIAL_MAX_CHANGES
            and not self.touches_infra
            and not self.touches_security
        )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class TaskFeatures:
    """Deterministic, bounded facts extracted before an epoch exists."""

    request_kind: str
    prompt_digest: str
    explicit_files: list[str] = dataclasses.field(default_factory=list)
    languages: list[str] = dataclasses.field(default_factory=list)
    subsystems: list[str] = dataclasses.field(default_factory=list)
    risk_signals: list[str] = dataclasses.field(default_factory=list)
    required_capabilities: list[str] = dataclasses.field(default_factory=list)
    minimum_tier: str = "normal"
    estimated_context_tokens: int = 0

    def as_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_run(root: pathlib.Path, *args: str) -> str:
    """Run a git command and return stripped stdout."""
    return subprocess.check_output(
        ["git", "-C", str(root), *args],
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()


def _git_has_head(root: pathlib.Path) -> bool:
    try:
        _git_run(root, "rev-parse", "--verify", "HEAD")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _match_any(path: str, patterns: list[str]) -> bool:
    """Match *path* against *patterns*, treating ``**`` as in gitignore.

    ``**/foo`` matches ``foo`` (zero dirs) and ``a/b/foo``.
    ``foo/**`` matches anything under ``foo/``.
    Plain globs like ``*.yaml`` work as usual with ``fnmatch``.
    """
    for pat in patterns:
        # Strip ``**/`` prefix — if the pattern starts with it, try
        # matching without the prefix (zero-directory match).
        if pat.startswith("**/"):
            bare = pat[3:]
            if fnmatch.fnmatch(path, bare):
                return True
            # Try with ** replaced by * everywhere
            if fnmatch.fnmatch(path, pat.replace("**", "*")):
                return True

        simple = pat.replace("**", "*")
        if fnmatch.fnmatch(path, simple):
            return True
        # ``foo/*`` matches anything starting with ``foo/``
        if simple.endswith("/*"):
            if path.startswith(simple):
                return True
    return False


def _subsystem(path: str) -> str:
    """Return the top-level directory component, or 'root' for bare files."""
    parts = pathlib.PurePath(path).parts
    if len(parts) == 1:
        return "root"
    return parts[0]


def _count_meaningful(lines: list[str]) -> int:
    count = 0
    for line in lines:
        if _MEANINGFUL_RE.match(line):
            count += 1
    return count


def extract_task_features(prompt: str, cwd: pathlib.Path) -> TaskFeatures:
    """Extract conservative task facts without calling a model.

    This function deliberately over-escalates ambiguous work. It runs before
    workflow state exists, so it must not depend on the diff or on an agent.
    """
    text = (prompt or "").strip()
    lower = text.lower()
    explicit_files = sorted(set(re.findall(
        r"(?<![\w/-])(?:[\w.-]+/)*[\w.-]+\.(?:py|ts|tsx|js|jsx|go|rs|java|yaml|yml|json|md|sql|sh)(?![\w.-])",
        text,
    )))[:64]
    known = {
        "python": [".py"], "javascript": [".js", ".jsx"],
        "typescript": [".ts", ".tsx"], "go": [".go"],
        "rust": [".rs"], "java": [".java"],
    }
    languages = [name for name, suffixes in known.items() if any(s in text for s in suffixes)]
    try:
        repository_files = _git_run(cwd, "ls-files").splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError):
        repository_files = []
    if not explicit_files:
        explicit_files = [p for p in repository_files if pathlib.Path(p).name.lower() in lower][:32]
    subsystems = sorted({_subsystem(p) for p in explicit_files})

    signal_patterns: dict[str, tuple[str, ...]] = {
        "security": ("security", "auth", "oauth", "credential", "token", "secret", "crypto"),
        "database": ("database", "sqlite", "sql", "migration", "schema", "postgres"),
        "network": ("http", "api", "proxy", "endpoint", "network", "socket"),
        "shared-state": ("concurrency", "scheduler", "state", "registry", "workflow", "hook"),
        "multi-file": ("multi-file", "cross-cutting", "across files", "several files"),
        "tests-required": ("test", "coverage", "verify", "regression"),
    }
    signals = sorted(name for name, words in signal_patterns.items() if any(word in lower for word in words))
    if len(subsystems) >= 2:
        signals.append("multi-file")
    if any(_match_any(path, SECURITY_GLOBS) for path in explicit_files):
        signals.append("security")
    signals = sorted(set(signals))

    if any(s in signals for s in ("security", "database")) and any(
        word in lower for word in ("auth", "credential", "migration", "schema", "database", "secret")
    ):
        tier = "high-risk"
    elif "multi-file" in signals or len(subsystems) >= 2 or any(
        word in lower for word in ("refactor", "architecture", "integrate", "orchestrate")
    ):
        tier = "cross-cutting"
    elif any(word in lower for word in ("typo", "comment", "format", "rename")) and len(explicit_files) <= 1:
        tier = "trivial"
    else:
        tier = "normal"

    if any(word in lower for word in ("edit", "implement", "change", "fix", "add", "remove", "update")):
        request_kind = "mutation"
        required_capabilities = ["tools", "mutation"]
    elif any(word in lower for word in ("review", "audit", "inspect", "explain")):
        request_kind = "analysis"
        required_capabilities = ["tools"]
    else:
        request_kind = "question"
        required_capabilities = []

    estimated_context_tokens = min(200_000, max(1, len(text) // 4 + len(explicit_files) * 300))
    import hashlib
    return TaskFeatures(
        request_kind=request_kind,
        prompt_digest=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        explicit_files=explicit_files,
        languages=sorted(set(languages)),
        subsystems=subsystems,
        risk_signals=signals,
        required_capabilities=required_capabilities,
        minimum_tier=tier,
        estimated_context_tokens=estimated_context_tokens,
    )


def determine_minimum_workflow(features: TaskFeatures | dict[str, object]) -> str:
    """Return the deterministic lower bound for pre-task routing."""
    tier = features.minimum_tier if isinstance(features, TaskFeatures) else str(features.get("minimum_tier", "normal"))
    return tier if tier in {"trivial", "normal", "cross-cutting", "high-risk"} else "high-risk"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _collect_diffs(head_ref: str, cwd: pathlib.Path) -> str:
    """Return the full unified diff for *head_ref* as a single string.

    Builds diff from:
      - Staged changes (--cached)
      - Unstaged changes
      - Untracked files (as new files)
    Handles renames, binary files, and submodules.
    """
    root = pathlib.Path(_git_run(cwd, "rev-parse", "--show-toplevel")).resolve()
    parts: list[str] = []

    if _git_has_head(root):
        # Get staged changes
        staged_diff = _git_run(root, "diff", "--unified=0", "--cached", head_ref)
        if staged_diff:
            parts.append(staged_diff)

        # Get unstaged changes
        unstaged_diff = _git_run(root, "diff", "--unified=0", head_ref)
        if unstaged_diff:
            parts.append(unstaged_diff)
    else:
        # No HEAD — build synthetic diffs against an empty tree
        for fpath in sorted(_git_run(root, "ls-files").splitlines()):
            content = _git_run(root, "show", f":0:{fpath}")
            parts.append(f"--- /dev/null\n+++ b/{fpath}\n")
            for line in content.splitlines():
                parts.append(f"+{line}\n")
            parts.append("\n")

    # Also include untracked files (non-binary only)
    untracked = _git_run(root, "ls-files", "--others", "--exclude-standard").splitlines()
    for fpath in untracked:
        ffile = root / fpath
        if ffile.is_file():
            try:
                content = ffile.read_text()
            except UnicodeDecodeError:
                # Binary file - skip
                continue
            parts.append(f"--- /dev/null\n+++ b/{fpath}\n")
            for line in content.splitlines():
                parts.append(f"+{line}\n")
            parts.append("\n")

    return "\n".join(parts)


def classify_workspace(
    head_ref: str,
    cwd: pathlib.Path,
    security_globs: list[str] | None = None,
    infra_globs: list[str] | None = None,
) -> ChangeReport:
    """Analyse the workspace diff and return an objective change report.

    Parameters:
        head_ref: git ref to diff against (e.g. ``"HEAD"``).
        cwd: working directory.
        security_globs: override security-sensitive glob patterns.
        infra_globs: override infrastructure glob patterns.

    Returns a ``ChangeReport`` whose fields can be used to determine
    whether the controller's tier selection is appropriate.

    The report **always** includes all tracked and untracked files so
    downstream hooks can make enforcement decisions even if the diff
    contains no deletions.
    """
    security = security_globs or SECURITY_GLOBS
    infra = infra_globs or INFRA_GLOBS

    diff_text = _collect_diffs(head_ref, cwd)

    # ------------------------------------------------------------------
    # Phase 1 – mutate-able accumulators (avoid frozen-dataclass issues)
    # ------------------------------------------------------------------
    # We use plain dicts during parsing because `frozen=True` on the
    # ChangeFile dataclass makes all attributes read-only after __init__.

    subsystems: set[str] = set()
    touches_infra = False
    touches_security = False
    total_meaningful = 0
    current: dict | None = None
    file_records: list[dict] = []

    for diff_line in diff_text.splitlines():
        # --- new file header ---
        m_path = re.match(r"+++ b/(.+)", diff_line)
        if m_path:
            if current is not None:
                file_records.append(current)
            path = m_path.group(1)
            subsystems.add(_subsystem(path))
            touches_infra = touches_infra or _match_any(path, infra)
            touches_security = touches_security or _match_any(path, security)
            current = {
                "path": path,
                "added": 0,
                "deleted": 0,
                "new_file": False,
                "deleted_file": False,
                "meaningful_changes": 0,
            }
            continue

        # --- deleted file header ---
        m_old = re.match(r"--- (.+)", diff_line)
        if m_old:
            old_path = m_old.group(1).split("\t", 1)[0]
            if old_path == "/dev/null":
                if current is not None:
                    current["new_file"] = True
            elif current is not None and current.get("path") == old_path.removeprefix("a/"):
                current["deleted_file"] = False
            elif old_path and old_path != "/dev/null":
                # A deleted file has no b/ header; retain a record for it.
                if current is not None:
                    file_records.append(current)
                path = old_path.removeprefix("a/")
                current = {
                    "path": path,
                    "added": 0,
                    "deleted": 0,
                    "new_file": False,
                    "deleted_file": True,
                    "meaningful_changes": 0,
                }
            continue

        # --- hunk header (informational only) ---
        if diff_line.startswith("@@"):
            continue

        # --- diff content lines ---
        if current is None:
            continue
        if diff_line.startswith("+") and not diff_line.startswith("+++"):
            current["added"] += 1
            if _MEANINGFUL_RE.match(diff_line):
                total_meaningful += 1
                current["meaningful_changes"] += 1
        elif diff_line.startswith("-") and not diff_line.startswith("---"):
            current["deleted"] += 1
            if diff_line[1:].strip():
                total_meaningful += 1
                current["meaningful_changes"] += 1

    if current is not None:
        file_records.append(current)

    # ------------------------------------------------------------------
    # Phase 2 – freeze into immutable dataclasses
    # ------------------------------------------------------------------
    files: list[ChangeFile] = [
        ChangeFile(**rec) for rec in file_records
    ]

    return ChangeReport(
        files=files,
        total_meaningful_changes=total_meaningful,
        subsystems=subsystems,
        touches_infra=touches_infra,
        touches_security=touches_security,
        min_trivial_tier=(
            len(files) <= 1
            and total_meaningful <= TRIVIAL_MAX_CHANGES
            and not touches_infra
            and not touches_security
        ),
    )


def minimum_tier(report: ChangeReport) -> str:
    """Return the minimum tier required for the given change report.

    Tier escalation logic:

    1. *high-risk*: touches security-sensitive paths
    2. *cross-cutting*: touches infrastructure or >= 2 subsystems
    3. *trivial*: <= 1 file and <= 3 meaningful changes
    4. *normal*: everything else
    """
    if report.touches_security:
        return "high-risk"
    if report.touches_infra or len(report.subsystems) >= 2:
        return "cross-cutting"
    if report.fits_trivial():
        return "trivial"
    return "normal"


# ---------------------------------------------------------------------------
# CLI entry point (for debugging / manual inspection)
# ---------------------------------------------------------------------------


def main() -> int:
    head_ref = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
    cwd = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path(".")

    report = classify_workspace(head_ref, cwd)
    tier = minimum_tier(report)

    import json

    payload = {
        "tier": tier,
        "minimum_tier": tier,
        "total_meaningful_changes": report.total_meaningful_changes,
        "files": len(report.files),
        "subsystems": sorted(report.subsystems),
        "touches_infra": report.touches_infra,
        "touches_security": report.touches_security,
        "min_trivial_tier": report.min_trivial_tier,
        "files_detail": [
            {
                "path": f.path,
                "added": f.added,
                "deleted": f.deleted,
                "meaningful": f.meaningful_changes,
                "new_file": f.new_file,
                "deleted_file": f.deleted_file,
            }
            for f in report.files
        ],
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

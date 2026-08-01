"""Tests for ``router.enhanced_router.policy`` tier classification."""
from __future__ import annotations

import dataclasses
import pytest

from enhanced_router.policy import (
    ChangeFile,
    ChangeReport,
    INFRA_GLOBS,
    SECURITY_GLOBS,
    _match_any,
    _subsystem,
    _MEANINGFUL_RE,
    minimum_tier,
)


# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------


def test_changefile_frozen():
    cf = ChangeFile(path="foo.py", added=1, deleted=0, new_file=False,
                    deleted_file=False, meaningful_changes=1)
    with pytest.raises(dataclasses.FrozenInstanceError):  # noqa: F821
        cf.added += 1  # type: ignore[assignment]


def test_changereport_defaults():
    cr = ChangeReport()
    assert cr.files == []
    assert cr.total_meaningful_changes == 0
    assert cr.subsystems == set()
    assert cr.touches_infra is False
    assert cr.touches_security is False
    assert cr.min_trivial_tier is True
    assert cr.fits_trivial() is True


# ---------------------------------------------------------------------------
# minimum_tier escalation logic
# ---------------------------------------------------------------------------


def test_high_risk_security():
    cr = ChangeReport(
        files=[ChangeFile(path="secrets/token.txt", added=5, deleted=0,
                          new_file=True, deleted_file=False, meaningful_changes=5)],
        total_meaningful_changes=5,
        subsystems={"secrets"},
        touches_security=True,
    )
    assert minimum_tier(cr) == "high-risk"


def test_cross_cutting_infra():
    cr = ChangeReport(
        files=[ChangeFile(path="router/foo.py", added=1, deleted=0,
                          new_file=False, deleted_file=False, meaningful_changes=1)],
        total_meaningful_changes=1,
        subsystems={"router"},
        touches_infra=True,
    )
    assert minimum_tier(cr) == "cross-cutting"


def test_cross_cutting_multi_subsystem():
    cr = ChangeReport(
        files=[
            ChangeFile(path="hooks/hook.py", added=1, deleted=0,
                       new_file=False, deleted_file=False, meaningful_changes=1),
            ChangeFile(path="router/route.py", added=1, deleted=0,
                       new_file=False, deleted_file=False, meaningful_changes=1),
        ],
        total_meaningful_changes=2,
        subsystems={"hooks", "router"},
        touches_infra=False,
    )
    assert minimum_tier(cr) == "cross-cutting"


def test_trivial_fits():
    cr = ChangeReport(
        files=[ChangeFile(path="app.py", added=2, deleted=1,
                          new_file=False, deleted_file=False, meaningful_changes=3)],
        total_meaningful_changes=3,
        subsystems={"root"},
        touches_infra=False,
        touches_security=False,
    )
    assert minimum_tier(cr) == "trivial"
    assert cr.fits_trivial() is True


def test_normal_exceeds_trivial_limits():
    cr = ChangeReport(
        files=[ChangeFile(path="app.py", added=10, deleted=5,
                          new_file=False, deleted_file=False, meaningful_changes=15)],
        total_meaningful_changes=15,
        subsystems={"root"},
        touches_infra=False,
        touches_security=False,
    )
    assert minimum_tier(cr) == "normal"
    assert cr.fits_trivial() is False


def test_two_files_exceeds_trivial():
    """Two files = not trivial, but single subsystem = normal (not cross-cutting)."""
    cr = ChangeReport(
        files=[
            ChangeFile(path="a.py", added=1, deleted=0,
                       new_file=False, deleted_file=False, meaningful_changes=1),
            ChangeFile(path="b.py", added=1, deleted=0,
                       new_file=False, deleted_file=False, meaningful_changes=1),
        ],
        total_meaningful_changes=2,
        subsystems={"root"},
        touches_infra=False,
        touches_security=False,
    )
    assert cr.fits_trivial() is False
    assert minimum_tier(cr) == "normal"


# ---------------------------------------------------------------------------
# _match_any glob tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path,expected", [
    ("router/foo.py", True),
    ("router/deep/bar.py", True),
    ("hooks/hook.py", True),
    (".claude/agents/x.md", True),
    ("agents/x.md", True),
    ("bin/script", True),
    ("install.sh", True),
    ("pyproject.toml", True),
    ("src/pyproject.toml", True),
    ("requirements.txt", True),
    ("config/settings.yaml", True),
    ("config/settings.yml", True),
    ("package.json", True),
    ("app.py", False),
    ("src/main.py", False),
    ("README.md", False),
])
def test_infra_match_any(path, expected):
    assert _match_any(path, INFRA_GLOBS) == expected


@pytest.mark.parametrize("path,expected", [
    ("secrets/token.txt", True),
    ("config/.env", True),
    ("providers.env", True),
    ("auth/login.py", True),
    ("crypto/keys.pem", True),
    ("tokens/secret.token", True),
    ("router.token", True),
    ("litellm.token", True),
    ("secrets/secret_key", True),
    ("credentials.json", True),
    ("app.py", False),
    ("router/route.py", False),
])
def test_security_match_any(path, expected):
    assert _match_any(path, SECURITY_GLOBS) == expected


# ---------------------------------------------------------------------------
# subsystem extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path,expected", [
    ("app.py", "root"),
    ("src/main.py", "src"),
    ("router/foo.py", "router"),
    ("hooks/hook.py", "hooks"),
    ("deep/nested/file.py", "deep"),
])
def test_subsystem(path, expected):
    assert _subsystem(path) == expected


# ---------------------------------------------------------------------------
# Meaningful change detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("line,matches", [
    ("+foo", True),
    ("+ bar", True),
    ("-foo", True),
    ("++something", False),  # starts with ++ (context line in diff)
    ("+", False),  # single + with nothing after = no match
    ("+", False),  # single + with nothing after = no match
    ("+++", False),  # file header
    ("---", False),  # file header
    (" @@", False),
    ("+import foo", True),
])
def test_meaningful_regex(line, matches):
    assert (_MEANINGFUL_RE.match(line) is not None) == matches


# ---------------------------------------------------------------------------
# Integration: import from hooks
# ---------------------------------------------------------------------------


def test_policy_importable_from_hooks():
    """Verify the policy module can be imported as ``enhanced_router.policy``."""
    from enhanced_router import policy
    assert hasattr(policy, "classify_workspace")
    assert hasattr(policy, "minimum_tier")
    assert hasattr(policy, "ChangeReport")
    assert hasattr(policy, "ChangeFile")

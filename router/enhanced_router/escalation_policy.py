"""Pure adaptive escalation decisions.

This module deliberately has no SQLite or registry dependencies.  It turns
observable task signals into a decision; the state repository persists and
applies that decision atomically.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field


TIERS = ("trivial", "normal", "cross-cutting", "high-risk")


@dataclass(frozen=True)
class EscalationInputs:
    current_tier: str
    minimum_tier: str | None = None
    failed_tests: int = 0
    unresolved_findings: int = 0
    missing_requirements: int = 0
    changed_files: int = 0
    changed_security_paths: bool = False
    repeated_repairs: int = 0
    provider_failures: int = 0
    evidence_incomplete: bool = False
    deadline_pressure: bool = False
    signals: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class EscalationDecision:
    should_escalate: bool
    from_tier: str
    to_tier: str
    reasons: tuple[str, ...]
    policy_digest: str


def evaluate_escalation(inputs: EscalationInputs) -> EscalationDecision:
    current = inputs.current_tier if inputs.current_tier in TIERS else "normal"
    index = TIERS.index(current)
    reasons: list[str] = []
    target = index
    if inputs.changed_security_paths:
        target = max(target, TIERS.index("high-risk"))
        reasons.append("security-sensitive paths changed")
    if inputs.minimum_tier in TIERS:
        minimum_index = TIERS.index(inputs.minimum_tier)
        if minimum_index > target:
            target = minimum_index
            reasons.append(f"workspace changes require {inputs.minimum_tier} workflow")
    if inputs.failed_tests > 0:
        target = max(target, min(index + 1, TIERS.index("high-risk")))
        reasons.append("deterministic test failure")
    if inputs.unresolved_findings > 0 or inputs.repeated_repairs > 0:
        target = max(target, min(index + 1, TIERS.index("high-risk")))
        reasons.append("review findings remain after repair")
    if inputs.missing_requirements > 0 or inputs.evidence_incomplete:
        target = max(target, TIERS.index("cross-cutting"))
        reasons.append("requirement coverage or evidence is incomplete")
    if inputs.provider_failures >= 2:
        reasons.append("provider failures require route/admission recovery")
    if inputs.deadline_pressure and index < TIERS.index("high-risk"):
        reasons.append("deadline pressure requires controller adjudication")
        target = max(target, min(index + 1, TIERS.index("high-risk")))
    payload = {
        "inputs": inputs.__dict__,
        "target": TIERS[target],
        "reasons": reasons,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=list).encode()).hexdigest()
    return EscalationDecision(
        should_escalate=target > index,
        from_tier=current,
        to_tier=TIERS[target],
        reasons=tuple(dict.fromkeys(reasons)),
        policy_digest=digest,
    )

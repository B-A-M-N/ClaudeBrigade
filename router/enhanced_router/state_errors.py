"""Shared exceptions used across several not-yet-extracted state.py sections
and their already-extracted repository mixins.

``WorkflowStateError`` is raised from many places (run lifecycle, shadow
workspace, agent execution lifecycle) that live in different modules across
the incremental state.py extraction. A single canonical definition here
-- imported by every module that raises or catches it -- keeps
``except WorkflowStateError`` working regardless of which module actually
raised the instance, instead of accidentally creating lookalike classes
that don't match each other under Python's identity-based exception
matching.
"""

from __future__ import annotations


class WorkflowStateError(Exception):
    """Raised when a workflow phase transition is invalid."""

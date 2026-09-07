"""Typing/runtime seam shared by the incremental RouteState repositories.

Each repository is mixed into ``RouteState`` and may call methods supplied by
another repository.  Python resolves those methods through the final MRO, but
the individual mixin class cannot see that host surface in isolation.  This
small base gives static analyzers an explicit host seam without changing the
runtime dispatch or hiding missing methods behind a broad per-file ignore.
"""

from __future__ import annotations

import sqlite3
from typing import Any


class RepositoryMixin:
    """Common host contract for a RouteState repository mixin."""

    def _new_conn(self) -> sqlite3.Connection:  # pragma: no cover - host override
        raise NotImplementedError

    def __getattr__(self, name: str) -> Any:
        # RouteState supplies cross-repository methods through normal MRO.  If
        # a method is genuinely absent, preserve Python's normal failure.
        raise AttributeError(name)

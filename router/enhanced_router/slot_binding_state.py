"""Immutable model-slot snapshots for each active workflow epoch."""

from __future__ import annotations

from enhanced_router.repository_base import RepositoryMixin

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Iterable


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def insert_slot_bindings(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    epoch_id: str,
    bindings: Iterable[dict[str, Any]],
    registry_hash: str,
) -> None:
    """Insert an epoch's slot projection inside its surrounding transaction."""
    now = _utcnow()
    for binding in bindings:
        conn.execute(
            """INSERT INTO slot_bindings
               (run_id, epoch_id, slot_name, public_alias, model_alias,
                logical_model_id, provider_id, endpoint_id,
                fallback_policy_json, registry_hash, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(run_id, epoch_id, slot_name) DO NOTHING""",
            (
                run_id,
                epoch_id,
                binding["slot"],
                binding["public_model_alias"],
                binding["model_alias"],
                binding["model_id"],
                binding.get("provider_id"),
                None if binding.get("endpoint", "auto") == "auto" else binding.get("endpoint"),
                json.dumps(binding.get("fallbacks") or [], separators=(",", ":")),
                registry_hash,
                now,
            ),
        )


class SlotBindingRepository(RepositoryMixin):
    """Read/write access to immutable per-epoch slot bindings."""

    def _new_conn(self) -> sqlite3.Connection:  # pragma: no cover
        raise NotImplementedError

    def snapshot_slot_bindings(
        self,
        run_id: str,
        epoch_id: str,
        bindings: Iterable[dict[str, Any]],
        registry_hash: str,
    ) -> list[dict[str, Any]]:
        conn = self._new_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            insert_slot_bindings(
                conn,
                run_id=run_id,
                epoch_id=epoch_id,
                bindings=bindings,
                registry_hash=registry_hash,
            )
            conn.commit()
            return self.get_slot_bindings(run_id, epoch_id)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_slot_bindings(self, run_id: str, epoch_id: str) -> list[dict[str, Any]]:
        conn = self._new_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM slot_bindings WHERE run_id=? AND epoch_id=? ORDER BY slot_name",
                (run_id, epoch_id),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_slot_binding(
        self, run_id: str, epoch_id: str, slot_name: str
    ) -> dict[str, Any] | None:
        """Return one immutable slot snapshot for runtime route resolution."""
        conn = self._new_conn()
        try:
            row = conn.execute(
                "SELECT * FROM slot_bindings "
                "WHERE run_id=? AND epoch_id=? AND slot_name=?",
                (run_id, epoch_id, slot_name),
            ).fetchone()
            return dict(row) if row is not None else None
        finally:
            conn.close()

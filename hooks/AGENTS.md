# Claude Code Hooks

## Purpose

This subtree owns the Claude Code lifecycle hooks that intake tasks, guard
tools and native spawns, record evidence, audit mutations, expose status, and
validate completion.

## Ownership

Hook-side guardrails and evidence projection. Durable authority remains in the
router SQLite state; JSONL and marker files are diagnostic/projection artifacts
and must not become a second source of truth.

## Local Contracts

- Hooks receive JSON on stdin and must emit valid Claude Code hook output on
  stdout. Diagnostics belong on stderr or the ledger.
- `guard_tool.py` fails closed for unauthorized native spawns, mutation tools,
  workspace leases, and read-only shell commands. Regex detection is only a
  warning layer; allowlist/sandbox and post-tool fingerprint evidence are the
  boundary.
- `audit_agent.py` correlates native lifecycle events to claimed actions and
  owned shadow workspaces. It must not commit, stash, reset, silently merge, or
  mutate the canonical checkout.
- `workspace_fingerprint.py` must recalculate mutable state for pre/post
  comparisons; do not add epoch-only caching.
- `completion_guard.py` fails closed when authoritative state cannot be read
  and validates state-generated completion evidence against the current
  workspace and route snapshot.
- Hooks may project state into session markers/statusline data, but resume must
  rehydrate from SQLite rather than deleting active state unconditionally.

## Work Guidance

Keep hook startup lightweight and imports resilient, but do not swallow an
authority failure as success. Use deterministic test doubles for router/state
calls. Any new hook field must be checked for missing, stale, replayed, and
cross-run values.

## Verification

```bash
pytest -q tests/test_ledger_and_guard.py tests/test_fingerprint.py
pytest -q tests/test_settings.py tests/test_launcher.py
pyright hooks/
```

## Child DOX Index

No child directories. Hook scripts are the local execution units.

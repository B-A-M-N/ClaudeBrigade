# Milestone Records

## Purpose

This subtree contains milestone handoffs and implementation records for the
ClaudeBrigade build history.

## Ownership

Planning and historical implementation documentation. Runtime behavior is
owned by `router/`, `hooks/`, `config/`, and `agents/`.

## Local Contracts

- Records must distinguish implemented behavior, planned work, and known
  limitations; do not use milestone completion as proof of current runtime
  correctness.
- Paths, commands, and architecture claims must point to current files or be
  clearly labeled historical.
- New architectural truth belongs in `ARCHITECTURE.md` and the relevant local
  `AGENTS.md`; milestone records may link to it rather than duplicate it.

## Work Guidance

Update a milestone record when its scope or verification evidence changes.
Preserve the original intent and date/context; correct stale technical claims
instead of adding contradictory notes.

## Verification

```bash
rg -n "TODO|Remaining|Implemented|Verification" meta/
git diff --check
```

## Child DOX Index

No child directories. `M01` through `M09` are milestone records.

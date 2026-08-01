# Operational Commands

## Purpose

This subtree contains the user-facing launcher, doctor, login, router-stop,
and milestone planning commands.

## Ownership

Process startup/shutdown, environment isolation, installation paths, runtime
diagnostics, and operator-facing shell behavior.

## Local Contracts

- `claude-brigade` owns the Claude Code agent bundle, controller prompt
  appendix, router credentials, MCP configuration, provider-key isolation,
  run ID, and router process lifecycle.
- Provider keys stay in the router process and must be removed from the
  Claude Code environment before launch.
- Token and provider files must remain owner-only and reject symlink attacks.
- User-supplied Claude flags that would replace the managed agent/system
  contract must be rejected; controller model selection is explicitly allowed.
- `claude-brigade-doctor` reports readiness and configuration problems; it must
  not claim a route is healthy merely because an alias is present.
- Do not add destructive cleanup, reset, stash, commit, or force-merge logic
  to hooks or launcher paths.

## Work Guidance

Preserve `set -euo pipefail`, restrictive `umask`, bounded port selection, and
explicit absolute paths. Test shell changes with the launcher test doubles;
never require live Claude or provider credentials in the unit suite.

## Verification

```bash
pytest -q tests/test_launcher.py tests/test_settings.py
bash -n bin/claude-brigade bin/claude-brigade-doctor bin/claude-brigade-login bin/claude-brigade-router-stop
```

## Child DOX Index

No child directories. Each executable is an operator command.

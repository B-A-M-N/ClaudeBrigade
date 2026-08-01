# Migration Guide: claude-enhanced → claude-brigade

## Overview

ClaudeBrigade replaces the old `claude-enhanced` naming and introduces the Brigade architecture — an identity-first routing system backed by a LiteLLM proxy, a workflow engine with risk-tiered agent orchestration, and hardened security primitives. Your normal `claude` command and `~/.claude` profile are untouched.

## Config path migration

Your config directory moved from `~/.config/claude-enhanced` to `~/.config/claude-brigade`. On first install, the setup script copies `models.yaml`, `profiles.yaml`, `workflows.yaml`, `router.token`, and `longcat.env` from the old location into the new one.

## Binary name changes

| Old name                          | New name                           |
|-----------------------------------|------------------------------------|
| `claude-enhanced`                 | `claude-brigade`                   |
| `claude-enhanced-doctor`          | `claude-brigade-doctor`            |
| `claude-enhanced-login`           | `claude-brigade-login`             |
| `claude-enhanced-router-stop`     | `claude-brigade-router-stop`       |

The installer still copies the old `claude-enhanced*` binaries for backwards compatibility. You should update your shell aliases and scripts to use the new names.

## Provider env file

The provider keys file was renamed from `longcat.env` to `providers.env`. On first install, the script extracts `LONGCAT_API_KEY` from your old `longcat.env` and writes it into `providers.env`. If you have a custom `providers.env`, no action is needed. Otherwise, edit `~/.config/claude-brigade/providers.env` after install.

## New config files

The config directory now expects three YAML files:

- `models.yaml` — model route definitions and tier assignments
- `profiles.yaml` — session profile configuration
- `workflows.yaml` — workflow engine definitions

The installer creates sensible defaults for any file that is missing. If you previously had custom config under `~/.config/claude-enhanced/config/`, those are copied during migration.

## Router token regeneration

The router token moved to `~/.config/claude-brigade/router.token`. The migration copies the old token if present. If the file is absent after install, the installer generates a new one automatically. After regenerating, stop the running router with `claude-brigade-router-stop` before your next launch.

## State DB compatibility

The SQLite state database lives at `~/.local/state/claude-brigade/state.db`. It
auto-migrates sequentially to the current schema version, including endpoint
configuration identity, provider catalog snapshots, execution telemetry,
workspace leases, and logical deployment-group policy. No manual migration is
required. Existing bindings remain pinned to their stored route evidence.

## Rollback instructions

To uninstall ClaudeBrigade and revert to the original setup:

1. Remove the Brigade profile and config:
   ```bash
   rm -rf ~/.claude-brigade ~/.config/claude-brigade ~/.cache/claude-brigade
   ```
2. Remove the Brigade binaries:
   ```bash
   rm -f ~/.local/bin/claude-brigade ~/.local/bin/claude-brigade-*
   ```
3. Restore any config files you backed up from `~/.config/claude-enhanced/`.
4. Restart your normal `claude` command.

Your old `~/.config/claude-enhanced/` directory is preserved by the migration (not overwritten), so a clean rollback path exists if needed.

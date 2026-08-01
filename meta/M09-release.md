# M09: Branding, Migration, Documentation, E2E

## Status
COMPLETE — all key artifacts delivered

## Key artifacts
- Migration guide: `MIGRATION.md` with full config path, binary name, env file, token, state DB, rollback instructions
- Documentation: `README.md` updated with LiteLLM, workflow engine, policy tier, config docs
- E2E tests: 30+ tests covering routing, binding lifecycle, MCP control, LiteLLM dispatch, workflow lifecycle, non-happy-path scenarios (503/409, drained gens, closed epochs, multi-agent)
- Binary rename: `bin/claude-brigade*` family with backward-compatible `claude-enhanced*` symlinks in install.sh
- `lock.py`, `env_parser.py` git-tracked
- `pyproject.toml` auto-discovers all modules
- SHA256SUMS regenerated and verifying

## Not implemented
- Release zip/tarball packaging script
- Performance benchmark suite
- Python version CI matrix
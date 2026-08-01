#!/usr/bin/env python3
"""Bootstrap environment loader for the ClaudeBrigade launcher.

This module provides a clean CLI entry point for the launcher to parse
providers.env using the hardened env_parser.py, avoiding bash-based parsing.

Usage:
    python -m enhanced_router.bootstrap_env --config-dir /path/to/config
    python -m enhanced_router.bootstrap_env --config-dir /path/to/config --emit-nul

Output (stdout):
    JSON object with provider_env and provider_keys fields.

Or with --emit-nul:
    NUL-delimited KEY=VALUE pairs for bash consumption
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure we can import from enhanced_router
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_router.env_parser import EnvParseError, parse_env_file


# Provider keys that are allowed in providers.env
ALLOWED_PROVIDER_KEYS = frozenset({
    "FREEINFERENCE_API_KEY",
    "FREEINFERENCE_API_BASE",
    "FREEINFERENCE_MAX_CONCURRENCY",
    "LONGCAT_API_KEY",
    "LONGCAT_API_BASE",
    "OPENROUTER_API_KEY",
    "DEEPSEEK_API_KEY",
    "GLM_API_KEY",
    "ANTHROPIC_API_KEY",
    "GROQ_API_KEY",
    "OPENAI_API_KEY",
    "MISTRAL_API_KEY",
    "CEREBRAS_API_KEY",
    "COHERE_API_KEY",
    "REPLICATE_API_KEY",
    "PALM_API_KEY",
    "VERTEX_PROJECT",
    "KILO_API_KEY",
    "CLINE_API_KEY",
    "CLINEPASS_API_KEY",
    "OPENCODE_API_KEY",
    "OPENCODEGO_API_KEY",
    "NVIDIA_API_KEY",
    "OLLAMA_API_KEY",
    "FREETHEAI_API_KEY",
    "REQUESTY_API_KEY",
    "FREEMODEL_API_KEY",
})

# Provider keys that should be passed to the router process (not Claude Code)
PROVIDER_KEYS_FOR_ROUTER = frozenset({
    "FREEINFERENCE_API_KEY",
    "FREEINFERENCE_API_BASE",
    "FREEINFERENCE_MAX_CONCURRENCY",
    "LONGCAT_API_KEY",
    "LONGCAT_API_BASE",
    "OPENROUTER_API_KEY",
    "DEEPSEEK_API_KEY",
    "GLM_API_KEY",
    "ANTHROPIC_API_KEY",
    "KILO_API_KEY",
    "CLINE_API_KEY",
    "CLINEPASS_API_KEY",
    "OPENCODE_API_KEY",
    "OPENCODEGO_API_KEY",
    "NVIDIA_API_KEY",
    "OLLAMA_API_KEY",
    "FREETHEAI_API_KEY",
    "REQUESTY_API_KEY",
    "FREEMODEL_API_KEY",
})

# Only API credentials belong in the OS keyring.  Non-secret tuning values
# such as FREEINFERENCE_MAX_CONCURRENCY remain ordinary configuration.
PROVIDER_SECRET_KEYS = frozenset(
    key for key in ALLOWED_PROVIDER_KEYS if key.endswith("_API_KEY")
)


@dataclass(frozen=True)
class BootstrapResult:
    """Validated provider environment and the keys safe for router startup."""

    provider_env: dict[str, str]
    provider_keys: tuple[str, ...]


def load_providers_env(config_dir: Path) -> BootstrapResult:
    """Load and validate providers.env using the hardened parser.

    Returns:
        A stable typed result. Missing providers.env is a valid empty result.
    """
    providers_file = config_dir / "providers.env"
    if not providers_file.exists():
        return BootstrapResult({}, ())

    # Parse with strict validation
    parsed = parse_env_file(providers_file, allowed_keys=ALLOWED_PROVIDER_KEYS)

    # Filter to keys the router needs
    router_env = {k: v for k, v in parsed.items() if k in PROVIDER_KEYS_FOR_ROUTER}
    provider_keys = list(router_env.keys())

    return BootstrapResult(router_env, tuple(provider_keys))


def load_router_credentials(config_dir: Path) -> BootstrapResult:
    """Load router credentials with the OS keyring taking precedence.

    ``providers.env`` is retained as a migration path for existing installs,
    but new credentials saved by the configuration CLI live in the OS
    keyring.  This function is called inside the router process, so keyring
    values never need to pass through the launcher shell or Claude Code.
    """
    file_result = load_providers_env(config_dir) if (config_dir / "providers.env").exists() else BootstrapResult({}, ())
    merged = dict(file_result.provider_env)
    keyring_keys: tuple[str, ...] = ()
    try:
        from enhanced_router.credential_store import all_materialized, resolve

        secure = {
            key: value
            for key in PROVIDER_SECRET_KEYS
            if (value := resolve(key, config_dir))
        }
        # LiteLLM receives generated slot-specific names inside the router
        # process so a model group can load several keys without exposing
        # them to Claude Code or the launcher shell.
        secure.update(all_materialized(config_dir, PROVIDER_SECRET_KEYS))
        merged.update(secure)
        keyring_keys = tuple(sorted(set(secure) & set(PROVIDER_SECRET_KEYS)))
    except Exception:
        # A missing desktop keyring must not make a legacy file-only install
        # unusable.  The CLI reports the fallback explicitly when saving.
        pass
    return BootstrapResult(merged, tuple(sorted(set(file_result.provider_keys) | set(keyring_keys))))


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Bootstrap environment from providers.env")
    parser.add_argument(
        "--config-dir",
        required=True,
        help="Path to BRIGADE_CONFIG_DIR (contains providers.env)",
    )
    parser.add_argument(
        "--emit-nul",
        action="store_true",
        help="Emit NUL-delimited KEY=VALUE pairs for bash consumption",
    )
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    if not config_dir.is_dir():
        print(f"ERROR: Config directory not found: {config_dir}", file=sys.stderr)
        return 1

    try:
        result = load_providers_env(config_dir)
    except EnvParseError as exc:
        print(f"ERROR: Failed to parse providers.env: {exc}", file=sys.stderr)
        return 1

    if args.emit_nul:
        # Emit NUL-delimited pairs for bash
        for key in sorted(result.provider_env):
            value = result.provider_env[key]
            sys.stdout.write(f"{key}={value}\x00")
        sys.stdout.flush()
    else:
        # Emit JSON for programmatic consumption
        output = {
            "provider_env": result.provider_env,
            "provider_keys": list(result.provider_keys),
        }
        json.dump(output, sys.stdout)
        sys.stdout.write("\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

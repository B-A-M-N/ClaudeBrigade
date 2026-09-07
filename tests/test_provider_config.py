from __future__ import annotations

from pathlib import Path

import yaml

from enhanced_router.provider_config import (
    PROVIDER_SCHEMA_VERSION,
    merge_missing_provider_defaults,
    migrate_provider_config,
)


def test_migration_adds_missing_discovery_without_overwriting_custom_fields(tmp_path: Path):
    user = {
        "providers": {
            "freeinference": {
                "display_name": "My FreeInference",
                "api_key_env": "CUSTOM_FREEINFERENCE_KEY",
                "endpoints": {"openai": "http://localhost:9999/v1"},
                "limits": {"max_concurrency": 17},
            },
            "removed-provider": {"display_name": "Keep Removed"},
            "custom-provider": {"display_name": "Keep Added"},
        }
    }
    bundled = {
        "providers": {
            "freeinference": {
                "display_name": "FreeInference",
                "api_key_env": "FREEINFERENCE_API_KEY",
                "max_concurrency_env": "FREEINFERENCE_MAX_CONCURRENCY",
                "discovery": {
                    "type": "openai-models",
                    "url": "https://freeinference.org/v1/models",
                    "endpoint_id": "openai",
                },
            }
        }
    }

    assert merge_missing_provider_defaults(user, bundled)
    provider = user["providers"]["freeinference"]
    assert provider["display_name"] == "My FreeInference"
    assert provider["api_key_env"] == "CUSTOM_FREEINFERENCE_KEY"
    assert provider["endpoints"]["openai"] == "http://localhost:9999/v1"
    assert provider["limits"] == {"max_concurrency": 17}
    assert provider["max_concurrency_env"] == "FREEINFERENCE_MAX_CONCURRENCY"
    assert provider["discovery"]["url"] == "https://freeinference.org/v1/models"
    assert "removed-provider" in user["providers"]
    assert "custom-provider" in user["providers"]
    assert user["provider_schema_version"] == PROVIDER_SCHEMA_VERSION


def test_migration_writes_versioned_user_file(tmp_path: Path):
    user_path = tmp_path / "providers.yaml"
    bundled_path = tmp_path / "providers.yaml.example"
    user_path.write_text(yaml.safe_dump({"providers": {"freeinference": {"display_name": "FI"}}}))
    bundled_path.write_text(yaml.safe_dump({
        "provider_schema_version": 2,
        "providers": {
            "freeinference": {
                "display_name": "FreeInference",
                "discovery": {
                    "type": "openai-models",
                    "url": "https://freeinference.org/v1/models",
                    "endpoint_id": "openai",
                },
            }
        },
    }))

    assert migrate_provider_config(user_path, bundled_path)
    migrated = yaml.safe_load(user_path.read_text())
    assert migrated["provider_schema_version"] == 2
    assert migrated["providers"]["freeinference"]["display_name"] == "FI"
    assert migrated["providers"]["freeinference"]["discovery"]["url"].endswith("/models")

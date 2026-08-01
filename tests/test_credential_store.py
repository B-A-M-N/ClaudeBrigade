from __future__ import annotations

import yaml

from enhanced_router import credential_store
from enhanced_router.litellm_config import generate_litellm_config
from enhanced_router.config_models import ModelCapabilities, ModelSpec


def test_loaded_credentials_rotate_between_named_slots(tmp_path, monkeypatch):
    (tmp_path / credential_store.SLOTS_FILE).write_text(
        yaml.safe_dump({
            "credentials": {
                "OPENROUTER_API_KEY": {
                    "slots": ["primary", "backup"],
                    "rotation": "round_robin",
                }
            }
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        credential_store.materialized_env_name("OPENROUTER_API_KEY", "primary"),
        "primary-secret",
    )
    monkeypatch.setenv(
        credential_store.materialized_env_name("OPENROUTER_API_KEY", "backup"),
        "backup-secret",
    )
    credential_store._ROTATION_INDEX.clear()

    assert credential_store.resolve_loaded("OPENROUTER_API_KEY", tmp_path) == "primary-secret"
    assert credential_store.resolve_loaded("OPENROUTER_API_KEY", tmp_path) == "backup-secret"
    assert credential_store.resolve_loaded("OPENROUTER_API_KEY", tmp_path) == "primary-secret"


def test_litellm_model_group_contains_each_loaded_key_slot(tmp_path, monkeypatch):
    key = "OPENROUTER_API_KEY"
    (tmp_path / credential_store.SLOTS_FILE).write_text(
        yaml.safe_dump({"credentials": {key: {"slots": ["primary", "backup"]}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv(credential_store.materialized_env_name(key, "primary"), "one")
    monkeypatch.setenv(credential_store.materialized_env_name(key, "backup"), "two")
    monkeypatch.setenv("BRIGADE_CONFIG_DIR", str(tmp_path))
    model = ModelSpec(
        display_name="Router model",
        backend="litellm",
        litellm_model="openai/router-model",
        api_key_env=key,
        capabilities=ModelCapabilities(openai_chat_completions=True),
    )

    payload = yaml.safe_load(generate_litellm_config({"router-model": model}))
    entries = payload["model_list"]
    assert len(entries) == 2
    assert {
        item["litellm_params"]["api_key"] for item in entries
    } == {
        f"os.environ/{credential_store.materialized_env_name(key, 'primary')}",
        f"os.environ/{credential_store.materialized_env_name(key, 'backup')}",
    }


def test_provider_free_catalog_rules_are_conservative():
    from types import SimpleNamespace
    from enhanced_router.registry import ModelRegistry

    assert ModelRegistry._catalog_entry_is_free(
        "requesty", SimpleNamespace(raw={"input_price": 0, "output_price": 0})
    )
    assert not ModelRegistry._catalog_entry_is_free(
        "requesty", SimpleNamespace(raw={"input_price": 0, "output_price": 0.01})
    )
    assert ModelRegistry._catalog_entry_is_free(
        "kilocode", SimpleNamespace(raw={"id": "model:free"})
    )
    assert ModelRegistry._catalog_entry_is_free(
        "kilocode", SimpleNamespace(raw={"id": "kilo-auto/free"})
    )
    assert ModelRegistry._catalog_entry_is_free(
        "kilocode", SimpleNamespace(raw={"pricing": {"input": 0, "output": 0}})
    )
    assert not ModelRegistry._catalog_entry_is_free(
        "kilocode", SimpleNamespace(raw={"id": "model"})
    )

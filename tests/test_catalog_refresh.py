"""Tests for provider catalog discovery/refresh -- registry split fetch/apply
and config_cli.refresh_catalogs' TTL cache, bounded concurrency, and
structured per-provider results.
"""

from __future__ import annotations

import json
import pathlib
import time
from unittest import mock

import pytest
import yaml

from enhanced_router.config_cli import refresh_catalogs
from enhanced_router.provider_discovery import DiscoveredEndpoint
from enhanced_router.registry import ModelRegistry


@pytest.fixture
def config_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    cfg = tmp_path / "config"
    cfg.mkdir()
    return cfg


def _write_config(cfg: pathlib.Path, filename: str, data: dict) -> None:
    (cfg / filename).write_text(yaml.safe_dump(data), encoding="utf-8")


def _entry(provider_id: str, model_id: str) -> DiscoveredEndpoint:
    """A discovered entry as ``fetch_discovered_entries`` would return it --
    already provider-qualified, since ``apply_discovered_entries`` expects
    that qualification to have already happened.
    """
    return DiscoveredEndpoint(
        provider_id=provider_id, model_id=model_id, endpoint_id="openai",
        display_name=None, context_tokens=None, max_output_tokens=None,
        capabilities={}, raw={"id": model_id},
        logical_model_id=f"{provider_id}/{model_id}",
    )


def _registry_with_providers(config_dir: pathlib.Path, provider_ids: list[str]) -> ModelRegistry:
    _write_config(config_dir, "models.yaml", {"models": {}})
    _write_config(config_dir, "providers.yaml", {
        "providers": {
            pid: {
                "display_name": pid,
                "api_key_env": f"{pid.upper()}_API_KEY",
                "endpoints": {"openai": f"https://{pid}.example.invalid/v1"},
                "discovery": {"type": "openai-models"},
            }
            for pid in provider_ids
        },
    })
    reg = ModelRegistry(config_dir)
    reg.load_models()
    reg.load_providers()
    return reg


class TestFetchDiscoveredEntries:
    def test_discovery_url_overrides_endpoint_lookup(self, config_dir, monkeypatch):
        reg = _registry_with_providers(config_dir, ["acme"])
        reg._providers["acme"] = reg._providers["acme"].model_copy(
            update={"discovery": {"type": "openai-models", "url": "https://catalog.acme.invalid/v1"}}
        )
        monkeypatch.setenv("ACME_API_KEY", "test-key")

        captured = {}

        def fake_discover(**kwargs):
            captured.update(kwargs)
            return []

        with mock.patch("enhanced_router.provider_discovery.discover_openai_models", side_effect=fake_discover):
            reg.fetch_discovered_entries("acme")

        assert captured["base_url"] == "https://catalog.acme.invalid/v1"

    def test_falls_back_to_endpoint_when_no_discovery_url(self, config_dir, monkeypatch):
        reg = _registry_with_providers(config_dir, ["acme"])
        monkeypatch.setenv("ACME_API_KEY", "test-key")

        captured = {}

        def fake_discover(**kwargs):
            captured.update(kwargs)
            return []

        with mock.patch("enhanced_router.provider_discovery.discover_openai_models", side_effect=fake_discover):
            reg.fetch_discovered_entries("acme")

        assert captured["base_url"] == "https://acme.example.invalid/v1"

    def test_fetch_does_not_mutate_registry_state(self, config_dir, monkeypatch):
        reg = _registry_with_providers(config_dir, ["acme"])
        monkeypatch.setenv("ACME_API_KEY", "test-key")
        with mock.patch(
            "enhanced_router.provider_discovery.discover_openai_models",
            return_value=[_entry("acme", "model-x")],
        ):
            reg.fetch_discovered_entries("acme")
        assert "acme/model-x" not in reg.models
        assert not (config_dir / "discovered_models.yaml").exists()


class TestApplyDiscoveredEntries:
    def test_publishes_entries_and_persists(self, config_dir):
        reg = _registry_with_providers(config_dir, ["acme"])
        digest, count = reg.apply_discovered_entries(
            [_entry("acme", "model-x")], provider_id="acme",
        )
        assert count == 1
        assert digest
        assert "acme/model-x" in reg.models
        assert (config_dir / "discovered_models.yaml").exists()


class TestRefreshCatalogs:
    def test_skips_provider_missing_credential(self, config_dir):
        reg = _registry_with_providers(config_dir, ["acme"])
        results = refresh_catalogs(reg)
        assert len(results) == 1
        assert results[0].provider_id == "acme"
        assert results[0].status == "skipped"
        assert "ACME_API_KEY" in results[0].detail

    def test_success_result_persists_and_reports_model_count(self, config_dir, monkeypatch):
        reg = _registry_with_providers(config_dir, ["acme"])
        monkeypatch.setenv("ACME_API_KEY", "test-key")
        with mock.patch.object(
            reg, "fetch_discovered_entries", return_value=[_entry("acme", "model-x")],
        ):
            results = refresh_catalogs(reg)
        assert len(results) == 1
        assert results[0].status == "success"
        assert results[0].model_count == 1
        assert results[0].provider_id == "acme"

    def test_isolated_provider_error_does_not_block_others(self, config_dir, monkeypatch):
        reg = _registry_with_providers(config_dir, ["broken", "healthy"])
        monkeypatch.setenv("BROKEN_API_KEY", "k1")
        monkeypatch.setenv("HEALTHY_API_KEY", "k2")

        def fake_fetch(provider_id, **kwargs):
            if provider_id == "broken":
                raise RuntimeError("connection refused")
            return [_entry("healthy", "model-y")]

        with mock.patch.object(reg, "fetch_discovered_entries", side_effect=fake_fetch):
            results = refresh_catalogs(reg)

        by_id = {r.provider_id: r for r in results}
        assert by_id["broken"].status == "error"
        assert by_id["healthy"].status == "success"
        assert by_id["healthy"].model_count == 1

    def test_recently_refreshed_provider_is_skipped_within_ttl(self, config_dir, monkeypatch):
        reg = _registry_with_providers(config_dir, ["acme"])
        monkeypatch.setenv("ACME_API_KEY", "test-key")
        (config_dir / "catalog_refresh_state.json").write_text(
            json.dumps({"acme": {"refreshed_at": time.time(), "digest": "old"}})
        )
        with mock.patch.object(reg, "fetch_discovered_entries") as mock_fetch:
            results = refresh_catalogs(reg, ttl_seconds=900)

        mock_fetch.assert_not_called()
        assert results[0].status == "cached"

    def test_force_bypasses_ttl_cache(self, config_dir, monkeypatch):
        reg = _registry_with_providers(config_dir, ["acme"])
        monkeypatch.setenv("ACME_API_KEY", "test-key")
        (config_dir / "catalog_refresh_state.json").write_text(
            json.dumps({"acme": {"refreshed_at": time.time(), "digest": "old"}})
        )
        with mock.patch.object(
            reg, "fetch_discovered_entries", return_value=[_entry("acme", "model-x")],
        ) as mock_fetch:
            results = refresh_catalogs(reg, ttl_seconds=900, force=True)

        mock_fetch.assert_called_once()
        assert results[0].status == "success"

    def test_expired_ttl_triggers_a_fresh_fetch(self, config_dir, monkeypatch):
        reg = _registry_with_providers(config_dir, ["acme"])
        monkeypatch.setenv("ACME_API_KEY", "test-key")
        (config_dir / "catalog_refresh_state.json").write_text(
            json.dumps({"acme": {"refreshed_at": time.time() - 1000, "digest": "old"}})
        )
        with mock.patch.object(
            reg, "fetch_discovered_entries", return_value=[_entry("acme", "model-x")],
        ) as mock_fetch:
            results = refresh_catalogs(reg, ttl_seconds=900)

        mock_fetch.assert_called_once()
        assert results[0].status == "success"

    def test_all_providers_fetched_concurrently_still_applied(self, config_dir, monkeypatch):
        """Concurrent fetches must not race on the shared discovered-catalog
        file -- every provider's models end up persisted regardless of fetch
        completion order.
        """
        provider_ids = [f"provider{i}" for i in range(6)]
        reg = _registry_with_providers(config_dir, provider_ids)
        for pid in provider_ids:
            monkeypatch.setenv(f"{pid.upper()}_API_KEY", "k")

        def fake_fetch(provider_id, **kwargs):
            return [_entry(provider_id, "model-x")]

        with mock.patch.object(reg, "fetch_discovered_entries", side_effect=fake_fetch):
            results = refresh_catalogs(reg)

        assert all(r.status == "success" for r in results)
        for pid in provider_ids:
            assert f"{pid}/model-x" in reg.models

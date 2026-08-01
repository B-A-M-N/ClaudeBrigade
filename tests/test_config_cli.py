"""Regression tests for the interactive configuration bootstrap."""

from __future__ import annotations

from pathlib import Path

import yaml

from enhanced_router.config_cli import _load_registry


def _write(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_load_registry_overlays_missing_bundled_compatibility_models(tmp_path: Path):
    _write(tmp_path / "models.yaml", {
        "models": {
            "legacy": {
                "display_name": "Legacy",
                "backend": "litellm",
                "litellm_model": "ollama/legacy",
                "capabilities": {"tools": True, "mutation": True},
                "allowed_roles": ["recon", "implementer", "adversary", "repairer"],
            },
        },
    })
    _write(tmp_path / "models.yaml.example", {
        "models": {
            "diffusiongemma": {
                "display_name": "DiffusionGemma",
                "provider_id": "freeinference",
                "backend": "litellm",
                "litellm_model": "openai/diffusiongemma",
                "capabilities": {"tools": False, "mutation": False},
                "allowed_roles": [],
            },
        },
    })
    _write(tmp_path / "profiles.yaml", {
        "profiles": {
            "legacy": {
                "recon": "legacy",
                "implementer": "legacy",
                "adversary": "legacy",
                "repairer": "legacy",
            },
        },
    })
    _write(tmp_path / "workflows.yaml", {"workflows": {"normal": {"default_profile": "legacy"}}})
    _write(tmp_path / "providers.yaml", {"providers": {}})
    _write(tmp_path / "fastpath.yaml", {
        "fastpath": {"enabled": True, "model_id": "diffusiongemma"},
    })
    _write(tmp_path / "sidecars.yaml", {"sidecars": {}})

    registry, _ = _load_registry(tmp_path)

    assert "diffusiongemma" in registry.models
    assert registry.fastpath is not None

from pathlib import Path

import yaml

from enhanced_router.sidecar_config import migrate_sidecar_config
from enhanced_router.config_models import SidecarProfileSpec


def test_sidecar_migration_adds_workers_without_overwriting_operator_entries(tmp_path: Path):
    user = tmp_path / "sidecars.yaml"
    bundled = tmp_path / "sidecars.yaml.example"
    user.write_text(
        yaml.safe_dump({
            "sidecars": {"legacy-review": {"model_id": "operator-model"}},
            "sidecar_agents": {"qwen-grounder": {"model_id": "operator-qwen"}},
        }),
        encoding="utf-8",
    )
    bundled.write_text(
        yaml.safe_dump({
            "sidecar_agents": {
                "qwen-grounder": {"model_id": "bundled-qwen"},
                "minimax-reviewer": {"model_id": "bundled-minimax"},
            },
            "coprocessors": {
                "route-advisor": {"model_id": "diffusiongemma"},
            },
        }),
        encoding="utf-8",
    )

    assert migrate_sidecar_config(user, bundled) is True
    merged = yaml.safe_load(user.read_text(encoding="utf-8"))
    assert merged["sidecars"]["legacy-review"]["model_id"] == "operator-model"
    assert merged["sidecar_agents"]["qwen-grounder"]["model_id"] == "operator-qwen"
    assert merged["sidecar_agents"]["minimax-reviewer"]["model_id"] == "bundled-minimax"
    assert merged["coprocessors"]["route-advisor"]["model_id"] == "diffusiongemma"


def test_disabled_profile_coprocessor_lane_is_explicit():
    profile = SidecarProfileSpec(
        coprocessors_enabled=False,
        coprocessor_ids=["reviewer"],
    )
    assert profile.coprocessors_enabled is False
    assert profile.coprocessor_ids == ["reviewer"]


def test_sidecar_migration_adds_global_coprocessor_switch_without_overwriting_operator_policy(tmp_path: Path):
    user = tmp_path / "sidecars.yaml"
    bundled = tmp_path / "sidecars.yaml.example"
    user.write_text(
        yaml.safe_dump({"coprocessors_enabled": False}),
        encoding="utf-8",
    )
    bundled.write_text(
        yaml.safe_dump({"coprocessors_enabled": True}),
        encoding="utf-8",
    )

    assert migrate_sidecar_config(user, bundled) is False
    assert yaml.safe_load(user.read_text(encoding="utf-8"))["coprocessors_enabled"] is False


def test_sidecar_migration_adds_global_coprocessor_switch_to_legacy_config(tmp_path: Path):
    user = tmp_path / "sidecars.yaml"
    bundled = tmp_path / "sidecars.yaml.example"
    user.write_text(yaml.safe_dump({"sidecars": {}}), encoding="utf-8")
    bundled.write_text(yaml.safe_dump({"coprocessors_enabled": True}), encoding="utf-8")

    assert migrate_sidecar_config(user, bundled) is True
    assert yaml.safe_load(user.read_text(encoding="utf-8"))["coprocessors_enabled"] is True

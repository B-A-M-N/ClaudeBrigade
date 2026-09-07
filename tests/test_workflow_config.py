from pathlib import Path

import yaml

from enhanced_router.workflow_config import migrate_workflow_config


def test_legacy_default_workflow_gains_package_graph_without_losing_edits(tmp_path: Path):
    user = tmp_path / "workflows.yaml"
    bundled = tmp_path / "workflows.yaml.example"
    user.write_text(yaml.safe_dump({
        "workflows": {
            "normal": {
                "default_profile": "operator-profile",
                "phases": [{
                    "id": "implementation",
                    "roles": ["implementer"],
                    "mutation": True,
                    "max_fanout": 3,
                }],
            },
        },
    }), encoding="utf-8")
    bundled.write_text(yaml.safe_dump({
        "workflows": {
            "normal": {
                "default_profile": "hybrid",
                "phases": [
                    {"id": "package-plan", "actor": "controller", "produces": "work_packages"},
                    {"id": "implementation", "roles": ["implementer"], "mutation": True,
                     "fanout_from": "work_packages", "completion_mode": "all_packages"},
                ],
            },
        },
    }), encoding="utf-8")

    assert migrate_workflow_config(user, bundled) is True
    migrated = yaml.safe_load(user.read_text(encoding="utf-8"))
    phases = migrated["workflows"]["normal"]["phases"]
    assert [item["id"] for item in phases] == ["package-plan", "implementation"]
    assert phases[1]["max_fanout"] == 3
    assert phases[1]["fanout_from"] == "work_packages"
    assert phases[1]["completion_mode"] == "all_packages"
    assert phases[1]["depends_on"] == ["package-plan"]
    assert migrated["workflows"]["normal"]["default_profile"] == "operator-profile"
    assert migrated["workflow_schema_version"] == 2


def test_custom_workflow_phase_graph_is_not_replaced(tmp_path: Path):
    user = tmp_path / "workflows.yaml"
    bundled = tmp_path / "workflows.yaml.example"
    user.write_text(yaml.safe_dump({
        "workflows": {"normal": {"default_profile": "custom", "phases": [
            {"id": "operator-phase", "actor": "controller"},
        ]}},
    }), encoding="utf-8")
    bundled.write_text(yaml.safe_dump({
        "workflows": {"normal": {"default_profile": "hybrid", "phases": [
            {"id": "implementation", "mutation": True},
        ]}},
    }), encoding="utf-8")

    assert migrate_workflow_config(user, bundled) is True
    migrated = yaml.safe_load(user.read_text(encoding="utf-8"))
    assert migrated["workflows"]["normal"]["phases"] == [{"id": "operator-phase", "actor": "controller"}]
    assert migrated["workflow_schema_version"] == 2

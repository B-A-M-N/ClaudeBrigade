from __future__ import annotations

import json
import pathlib
import sys
import argparse
from copy import deepcopy
from typing import Any

import yaml


SCALAR_OR_LIST_FIELDS = {
    "tools",
    "disallowedTools",
    "skills",
}


def _split_top_level_csv(value: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(value):
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            item = value[start:index].strip()
            if item:
                parts.append(item)
            start = index + 1
    final = value[start:].strip()
    if final:
        parts.append(final)
    return parts


def parse_agent(path: pathlib.Path) -> tuple[str, dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"{path}: missing YAML frontmatter")
    try:
        frontmatter_text, prompt = text[4:].split("\n---\n", 1)
    except ValueError as exc:
        raise ValueError(f"{path}: unterminated YAML frontmatter") from exc

    frontmatter = yaml.safe_load(frontmatter_text)
    if not isinstance(frontmatter, dict):
        raise ValueError(f"{path}: frontmatter must be an object")

    name = frontmatter.pop("name", None)
    description = frontmatter.get("description")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{path}: non-empty name is required")
    if not isinstance(description, str) or not description:
        raise ValueError(f"{path}: non-empty description is required")

    for field in SCALAR_OR_LIST_FIELDS:
        value = frontmatter.get(field)
        if isinstance(value, str):
            frontmatter[field] = _split_top_level_csv(value)

    frontmatter["prompt"] = prompt.strip()
    return name, frontmatter


def render_agents(
    directory: pathlib.Path,
    registry: Any | None = None,
    profile_id: str | None = None,
    sidecar_profile_id: str | None = None,
) -> dict[str, dict[str, Any]]:
    agents: dict[str, dict[str, Any]] = {}
    paths = sorted(directory.glob("*.md"))
    if not paths:
        raise ValueError(f"No agent definitions found in {directory}")
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---\n"):
            continue
        name, definition = parse_agent(path)
        if name in agents:
            raise ValueError(f"Duplicate agent name: {name}")
        agents[name] = definition
    REQUIRED_ROLES = {"brigade-recon", "brigade-implementer", "brigade-adversary", "brigade-repairer", "controller-direct"}
    EXPECTED_MODELS = {
        "brigade-recon": "anthropic-brigade-recon",
        "brigade-implementer": "anthropic-brigade-implementer",
        "brigade-adversary": "anthropic-brigade-adversary",
        "brigade-repairer": "anthropic-brigade-repairer",
    }
    NATIVE_MODELS = {
        "brigade-recon": "haiku",
        "brigade-implementer": "sonnet",
        "brigade-adversary": "opus",
        "brigade-repairer": "opus",
    }
    missing = REQUIRED_ROLES - set(agents.keys())
    if missing:
        raise ValueError(f"Missing required agent roles: {', '.join(sorted(missing))}")
    slot_projection = bool(
        registry is not None
        and profile_id
        and hasattr(registry, "slot_alias_manifest")
        and registry.slot_alias_manifest(profile_id)
    )
    expected_models = NATIVE_MODELS if slot_projection else EXPECTED_MODELS
    for agent_name, expected_model in expected_models.items():
        actual_model = agents[agent_name].get("model", "")
        # Checked-in role templates and installed callers may still use the
        # old public role alias. Accept it as input, but selected profiles
        # always project the native Claude slot below.
        if actual_model in {expected_model, EXPECTED_MODELS[agent_name], NATIVE_MODELS[agent_name]}:
            continue
        if actual_model != expected_model:
            raise ValueError(
                f"Agent '{agent_name}' has model '{actual_model}' but expected '{expected_model}'"
            )
    if registry is not None:
        if hasattr(registry, "native_worker_manifest"):
            manifest = registry.native_worker_manifest(profile_id, sidecar_profile_id)
        else:
            # Source-compatible test doubles and older embedded callers.
            manifest = registry.specialist_manifest(profile_id)
        slot_models = {
            "brigade-recon": "haiku",
            "brigade-implementer": "sonnet",
            "brigade-adversary": "opus",
            "brigade-repairer": "opus",
        }
        if slot_projection:
            for name, model_alias in slot_models.items():
                if name in agents:
                    agents[name]["model"] = model_alias
        for entry in manifest.values():
            native_name = entry["native_agent_name"]
            if native_name in agents:
                if slot_projection and entry.get("model_alias"):
                    agents[native_name]["model"] = entry["model_alias"]
                continue
            roles = list(entry.get("roles") or [entry.get("role", "recon")])
            base_name = str(entry.get("template") or f"brigade-{roles[0]}")
            if base_name not in agents:
                raise ValueError(
                    f"Cannot generate '{native_name}': missing base agent '{base_name}'"
                )
            generated = deepcopy(agents[base_name])
            if entry.get("description"):
                generated["description"] = str(entry["description"])
            else:
                source_id = entry.get("source_id") or entry.get("profile_id") or "active"
                generated["description"] = (
                    f"{generated['description']} Backed by {entry.get('model_id', '')} "
                    f"from the active {source_id} manifest."
                )
            generated["model"] = entry.get("model_alias") or entry["public_model_alias"]
            if entry.get("system_prompt"):
                generated["prompt"] = f"{entry['system_prompt']}\n\n{generated['prompt']}"
            if entry.get("tools"):
                generated["tools"] = list(entry["tools"])
            if entry.get("disallowed_tools"):
                generated["disallowedTools"] = list(entry["disallowed_tools"])
            if entry.get("max_turns"):
                generated["maxTurns"] = int(entry["max_turns"])
            if entry.get("effort"):
                generated["effort"] = entry["effort"]
            generated["background"] = bool(entry.get("background", True))
            if entry.get("can_mutate"):
                generated["isolation"] = "worktree"
                generated["permissionMode"] = entry.get("permission_mode") or "acceptEdits"
            else:
                generated.pop("isolation", None)
                generated.pop("permissionMode", None)
                generated["tools"] = [
                    tool for tool in generated.get("tools", [])
                    if tool not in {"Edit", "Write", "NotebookEdit"}
                ]
            agents[native_name] = generated
    return agents


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m enhanced_router.agents_json",
        description="Render Claude Code's native agent manifest.",
    )
    parser.add_argument("agent_directory", type=pathlib.Path)
    # Keep positional arguments readable for older installed launchers while
    # making the two independent profile lanes unambiguous for new callers.
    parser.add_argument("legacy_inference_profile", nargs="?", default=None)
    parser.add_argument("legacy_sidecar_profile", nargs="?", default=None)
    parser.add_argument("--inference-profile", dest="inference_profile", default=None)
    parser.add_argument("--sidecar-profile", dest="sidecar_profile", default=None)
    args = parser.parse_args(argv)
    if args.inference_profile and args.legacy_inference_profile:
        parser.error("inference profile supplied both positionally and by option")
    if args.sidecar_profile and args.legacy_sidecar_profile:
        parser.error("sidecar profile supplied both positionally and by option")
    args.profile_id = args.inference_profile or args.legacy_inference_profile or None
    args.sidecar_profile_id = args.sidecar_profile or args.legacy_sidecar_profile or None
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2
    try:
        from enhanced_router.registry import get_registry

        agents = render_agents(
            args.agent_directory,
            get_registry(),
            args.profile_id,
            args.sidecar_profile_id,
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(agents, separators=(",", ":"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import pathlib
import sys
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


def render_agents(directory: pathlib.Path) -> dict[str, dict[str, Any]]:
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
    if "enhanced-controller" not in agents:
        raise ValueError("enhanced-controller definition is missing")
    return agents


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m enhanced_router.agents_json AGENT_DIRECTORY", file=sys.stderr)
        return 2
    try:
        agents = render_agents(pathlib.Path(sys.argv[1]))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(agents, separators=(",", ":"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

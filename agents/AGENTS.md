# Native Agent Definitions

## Purpose

This subtree contains Claude Code agent definitions and the controller prompt
appendix injected by the launcher. These files define visible tool access,
isolation, role behavior, and the prompt contract for native executions.

## Ownership

Claude Code launch/profile surface, with routing identity supplied by the
registry and lifecycle authority supplied by the router/hooks.

## Local Contracts

- YAML frontmatter must parse through `enhanced_router.agents_json`.
- Stable role agents are `brigade-recon`, `brigade-implementer`,
  `brigade-adversary`, `brigade-repairer`, and `controller-direct`.
- Read-only agents must not receive mutation tools. Mutators must request
  `isolation: worktree`; the router still verifies the actual child workspace.
- The `model` field is a public router alias, not provider infrastructure
  identity. Model-qualified identities declared in profile registry metadata
  may be projected from a stable role template at launch.
- `controller-append.md` must preserve the cooperative MCP claim/spawn
  protocol and must not promise hidden fan-out or untracked authority.

## Work Guidance

Keep prompts concrete about scope, evidence, and return format. When changing
tools, isolation, or model aliases, update guard/lifecycle tests and the launch
manifest path. Do not embed provider keys, endpoint URLs, or mutable workflow
authority in an agent definition.

## Verification

```bash
pytest -q tests/test_agents_json.py tests/test_launcher.py
PYTHONPATH=router python3 -m enhanced_router.agents_json agents
```

## Child DOX Index

No child directories. Each Markdown file is a role definition owned here.

# Claude Enhanced: Sonnet Controller + Native LongCat Subagents

This installs a second command, `claude-enhanced`, backed by a separate Claude Code profile. Your normal `claude` command and `~/.claude` profile are not edited.

## What you get

```text
ordinary task
  -> Sonnet 5 main thread: enhanced-controller
     -> native Agent(longcat-recon) when repository-wide evidence is needed
     -> native Agent(longcat-implementer) for default source mutation
     -> Sonnet reviews the stable diff
     -> fresh native Agent(longcat-adversary) when risk warrants it
     -> Sonnet adjudicates findings
     -> native Agent(longcat-repairer) fixes accepted defects
     -> Sonnet runs final verification and a workspace-hash gate

Claude Code -> 127.0.0.1:8787 loopback router
  Sonnet/Claude model IDs       -> api.anthropic.com with subscription OAuth forwarded
  anthropic-longcat-2-0 alias   -> api.longcat.chat/anthropic with only the LongCat key
```

You give it normal requests. Sonnet chooses the tier and invokes native subagents itself.

The launcher reserves Claude Code flags that would replace the agent definitions, controller prompt, model route, or settings source. Use the normal `claude` command for an intentionally customized one-off session.

## Install

```bash
unzip claude-enhanced.zip
cd claude-enhanced
./install.sh
```

Edit the protected key file:

```bash
nano ~/.config/claude-enhanced/longcat.env
```

Use this format:

```bash
LONGCAT_API_KEY='your_key_here'
```

Authenticate the separate profile with the same Claude subscription account:

```bash
claude-enhanced-login
claude-enhanced-doctor
```

Then use it like ordinary Claude Code:

```bash
cd /path/to/repository
claude-enhanced
```

## Workflow behavior

- **Analysis-only request:** Sonnet answers directly. No fake implementation ceremony.
- **Trivial safe edit:** controlled `sonnet-direct` path.
- **Normal edit:** Sonnet contract -> LongCat implementation -> Sonnet diff review and verification.
- **Cross-cutting edit:** LongCat recon -> Sonnet contract -> LongCat implementation -> fresh LongCat adversary -> Sonnet adjudication -> LongCat repair -> final verification.
- **High-risk edit:** design adversary before implementation plus a different fresh implementation adversary afterward.

Subagents receive fresh contexts. The controller must pass explicit repository evidence, constraints, acceptance criteria, and test requirements into every native `Agent(...)` call.

## Enforced controls

- Only the five named native subagents are spawnable from the main controller.
- All six definitions are injected with native `--agents` session configuration, which outranks project-level agent files and prevents accidental name collisions or role replacement.
- Background subagents and conversation forks are disabled, so mutation is sequential.
- Only `longcat-implementer`, `longcat-repairer`, and the exceptional `sonnet-direct` role may use file-write tools.
- Read-only roles are blocked from obvious mutating shell commands. This is a guardrail, not an operating-system sandbox.
- Agent start/stop lifecycle evidence is recorded without prompts or source content.
- Cross-cutting completion requires recon plus one completed fresh adversary.
- High-risk completion requires recon plus two separately spawned adversaries.
- The final completion hook checks role lifecycles, unresolved findings, `git diff --check`, and a SHA-256 fingerprint of tracked changes plus untracked files.

## Isolation

- Normal Claude profile: `~/.claude`
- Enhanced Claude profile: `~/.claude-enhanced`
- LongCat key: `~/.config/claude-enhanced/longcat.env` with mode `0600`
- Router token: `~/.config/claude-enhanced/router.token` with mode `0600`
- Router log: `~/.cache/claude-enhanced/router.log`

The launcher explicitly removes inherited API-key, OAuth-token, cloud-provider, model-override, and global-subagent-model environment variables before starting Claude. This prevents a shell-level override from silently replacing subscription Sonnet or forcing every subagent onto the wrong model.

Because the profile is intentionally separate, user-level skills, plugins, and memory stored only under `~/.claude` are not automatically copied. Project-level `.claude` configuration and `CLAUDE.md` files still load normally.

## Router behavior

The router preserves Claude-bound headers and request fields, including the subscription OAuth capability header. For LongCat-bound calls it:

- rewrites `anthropic-longcat-2-0` to `LongCat-2.0`;
- removes the incoming Claude credential and injects the LongCat Bearer key;
- strips Claude-only context-management, output-config, prompt-cache, and beta tool-schema fields;
- converts adaptive thinking to LongCat's enabled/disabled form;
- streams upstream SSE without buffering;
- logs only route, model, session/agent IDs, HTTP status, and connection latency.

The optional LongCat token-count endpoint is not proxied; Claude Code estimates context usage locally.

## Diagnostics

```bash
claude-enhanced-doctor
cat ~/.cache/claude-enhanced/router.log
CLAUDE_CONFIG_DIR=~/.claude-enhanced claude doctor
CLAUDE_CONFIG_DIR=~/.claude-enhanced claude auth status --text
claude-enhanced-router-stop
```

After changing the LongCat key, run `claude-enhanced-router-stop` before the next launch so the router restarts with the new key.

## Important limitation

This is a practical compatibility layer, not an Anthropic-supported non-Claude configuration. LongCat documents Claude Code and Anthropic-format compatibility, but Claude Code evolves its request protocol over time. The router deliberately passes Anthropic traffic unchanged and normalizes only LongCat traffic, but future Claude Code or LongCat API changes may require a small compatibility update.

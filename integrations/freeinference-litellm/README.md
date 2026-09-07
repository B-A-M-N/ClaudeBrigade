# FreeInference LiteLLM Kit

Local, user-owned BYOK integration for FreeInference. The kit discovers the
models available to the user's key, generates a pinned LiteLLM configuration,
and exposes loopback-only OpenAI-compatible and Anthropic-compatible endpoints.

The FreeInference key is used only by the local LiteLLM process. Clients use a
separate local `LITELLM_MASTER_KEY`.

## Quick start

```bash
uv sync
cp .env.example .env
# edit .env and set FREEINFERENCE_API_KEY

uv run fi-litellm init
uv run fi-litellm sync
uv run fi-litellm run
```

The proxy binds to `127.0.0.1:4000` by default. Keep it loopback-only during
development. `fi-litellm doctor` performs only model discovery; use
`fi-litellm doctor --live` for an explicit synthetic inference probe.

## Claude Code

With the proxy running, use a per-process environment override:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:4000 \
ANTHROPIC_AUTH_TOKEN="$LITELLM_MASTER_KEY" \
ANTHROPIC_MODEL=glm-5.1 \
ANTHROPIC_SMALL_FAST_MODEL=glm-5.1 \
API_TIMEOUT_MS=600000 \
claude
```

Do not put `FREEINFERENCE_API_KEY` in Claude Code's environment. Do not claim
agent compatibility from a one-shot request. Run the contract harness and a
real tool loop for each model you intend to use. The harness checks
non-streaming and SSE transport, Anthropic compatibility, single and parallel
tool calls, tool-result continuation, structured output, usage accounting, and
early stream cancellation. A passing report is route/protocol evidence, not a
claim that the model can complete an arbitrary workflow.

## Commands

```text
fi-litellm init
fi-litellm sync
fi-litellm run
fi-litellm doctor [--live]
fi-litellm test MODEL
fi-litellm report
```

The harness writes sanitized JSON reports under `reports/`; prompts, tool
arguments, responses, and credentials are never written.

ClaudeBrigade can publish an explicitly reviewed report into its
route-qualified endpoint certification state with
`enhanced_router.certification.publish_contract_report`. This is a separate
operator action: running `fi-litellm test` or receiving a successful response
does not automatically certify a route.

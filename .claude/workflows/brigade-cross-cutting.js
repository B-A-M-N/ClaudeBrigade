/*
 * ClaudeBrigade dynamic-workflow driver.
 *
 * MCP remains the scheduler.  This script never invents work packages or
 * chooses a model: the scheduler agent reads the persisted action wave and
 * the worker agents must claim the exact action before doing anything.
 * Keep the fan-out slice at three; provider admission is still authoritative
 * and may return fewer actions.
 *
 * Dynamic workflow agents do not expose the Claude Code Agent-tool
 * `subagent_type`/spawn metadata to this script.  Consequently this driver
 * is intentionally limited to read-only recon/review waves.  Mutating native
 * actions continue through the controller's claimed Agent-tool path, where
 * WorktreeCreate and exact spawn attachment are enforceable.
 */
export const meta = {
  name: 'brigade-cross-cutting',
  description: 'Run an MCP-admitted, bounded read-only cross-cutting audit wave',
  phases: [
    { title: 'Admit', detail: 'read the persisted Brigade action wave' },
    { title: 'Inspect', detail: 'run at most three independent read-only workers' },
    { title: 'Report', detail: 'return evidence to the controller; never authorize completion' },
  ],
}

const runId = args?.run_id ?? args?.runId
const epochId = args?.epoch_id ?? args?.epochId
const scope = args?.scope ?? 'cross-cutting'

phase('Admit')
const wave = await agent(`
You are the ClaudeBrigade workflow admission reader. Use the authenticated
ClaudeBrigade MCP control tools to call get_runnable_action_wave for run
${runId ?? '<current run>'}, epoch ${epochId ?? '<current epoch>'}, with a
maximum of 3 actions. Do not claim actions, spawn agents, edit files, run
mutating commands, or make completion decisions. Return only JSON with an
actions array. Include each action's action_id, native_agent_name, worker_kind,
agent_id, native_slot, expected_model_alias, can_mutate, package_id, prompt,
and capability_digest. If MCP is unavailable, return {"actions":[]} and an
error field.
`, {
  label: 'brigade-admission-reader',
  phase: 'Admit',
  model: 'haiku',
  schema: {
    type: 'object',
    required: ['actions'],
    properties: {
      actions: { type: 'array', items: { type: 'object' } },
      error: { type: 'string' },
    },
  },
})

const actions = Array.isArray(wave?.actions)
  ? wave.actions.filter(action => action && action.can_mutate !== true).slice(0, 3)
  : []

phase('Inspect')
const results = await parallel(actions.map(action => () => agent(`
You are a read-only ClaudeBrigade workflow worker for an already admitted
action. This is an evidence-gathering pass, not an implementation.

Before inspecting anything, call the authenticated MCP tool
claim_runnable_action with action_id ${action.action_id}. If the claim fails,
stop and report the denial. If it succeeds, use the exact returned contract,
native agent name, package, and capability snapshot below. Do not call Agent
or Workflow, do not edit/write files, do not integrate, and do not claim
completion. Inspect the repository and return reproducible findings and tests
that the main controller can use.

Admitted action:
${JSON.stringify(action)}

Scope: ${scope}
`, {
  label: `brigade-readonly-${action.action_id}`,
  phase: 'Inspect',
  model: action.native_slot ?? 'haiku',
}))

phase('Report')
return {
  run_id: runId,
  epoch_id: epochId,
  admitted: actions.length,
  results,
  completion_authority: 'controller-and-mcp-only',
}

/*
 * ClaudeBrigade high-risk read-only workflow driver.
 *
 * The script is an execution driver only.  SQLite/MCP owns readiness,
 * claims, provider admission, evidence acceptance, and completion.  Native
 * mutating actions are deliberately returned to the controller's exact
 * Agent-tool path until the workflow runtime can carry Brigade spawn
 * correlation and WorktreeCreate metadata.
 */
export const meta = {
  name: 'brigade-high-risk',
  description: 'Run a bounded MCP-admitted high-risk design and verification wave',
  phases: [
    { title: 'Admit', detail: 'read critical actions from Brigade' },
    { title: 'Challenge', detail: 'run independent read-only investigators' },
    { title: 'Report', detail: 'return evidence for controller adjudication' },
  ],
}

const runId = args?.run_id ?? args?.runId
const epochId = args?.epoch_id ?? args?.epochId

phase('Admit')
const wave = await agent(`
Use ClaudeBrigade MCP get_runnable_action_wave for run ${runId ?? '<current run>'}
and epoch ${epochId ?? '<current epoch>'}, limit 3. Read the persisted phase
graph and return JSON {"actions": [...]} containing only read-only native
actions (can_mutate=false), preserving action_id, native_agent_name,
native_slot, expected_model_alias, package_id, prompt, and capability_digest.
Do not claim, spawn, edit, integrate, or authorize anything.
`, {
  label: 'brigade-critical-admission-reader',
  phase: 'Admit',
  model: 'fable',
  schema: {
    type: 'object',
    required: ['actions'],
    properties: { actions: { type: 'array', items: { type: 'object' } } },
  },
})

const actions = Array.isArray(wave?.actions)
  ? wave.actions.filter(action => action && action.can_mutate !== true).slice(0, 3)
  : []

phase('Challenge')
const results = await parallel(actions.map(action => () => agent(`
Perform an independent, read-only high-risk challenge for the exact
ClaudeBrigade action below. First call MCP claim_runnable_action for
${action.action_id}; stop if it is denied. Do not call Agent or Workflow and do
not modify files. Check security, state transitions, concurrency, endpoint and
credential boundaries, and the original acceptance contract. Return only
reproducible findings and evidence for the main controller.

${JSON.stringify(action)}
`, {
  label: `brigade-critical-${action.action_id}`,
  phase: 'Challenge',
  model: action.native_slot ?? 'fable',
}))

phase('Report')
return {
  run_id: runId,
  epoch_id: epochId,
  admitted: actions.length,
  results,
  completion_authority: 'controller-and-mcp-only',
}

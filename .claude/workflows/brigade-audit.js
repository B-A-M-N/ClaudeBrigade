/*
 * ClaudeBrigade final audit driver.  It is deliberately read-only and starts
 * no more than three agents.  Findings go back to the controller; this file
 * cannot satisfy a phase, accept a finding, integrate a changeset, or mark a
 * run complete.
 */
export const meta = {
  name: 'brigade-audit',
  description: 'Run an MCP-admitted bounded final evidence audit',
  phases: [
    { title: 'Admit', detail: 'read final audit actions from Brigade' },
    { title: 'Audit', detail: 'independently inspect the final workspace' },
    { title: 'Report', detail: 'return omissions and evidence gaps' },
  ],
}

const runId = args?.run_id ?? args?.runId
const epochId = args?.epoch_id ?? args?.epochId

phase('Admit')
const wave = await agent(`
Call ClaudeBrigade MCP get_runnable_action_wave for run ${runId ?? '<current run>'}
and epoch ${epochId ?? '<current epoch>'}, limit 3. Return JSON with the exact
read-only audit actions. Do not claim actions or change any state. If there
are no admitted actions, return an empty actions array.
`, {
  label: 'brigade-audit-admission-reader',
  phase: 'Admit',
  model: 'haiku',
  schema: {
    type: 'object',
    required: ['actions'],
    properties: { actions: { type: 'array', items: { type: 'object' } } },
  },
})

const actions = Array.isArray(wave?.actions)
  ? wave.actions.filter(action => action && action.can_mutate !== true).slice(0, 3)
  : []

phase('Audit')
const results = await parallel(actions.map(action => () => agent(`
Claim the exact read-only ClaudeBrigade action ${action.action_id} through MCP,
then audit the final workspace against its original contract and acceptance
criteria. Do not edit, write, spawn, integrate, or declare completion. Report
only reproducible omissions, stale evidence, unresolved findings, and the
tests or observations that support each result.

${JSON.stringify(action)}
`, {
  label: `brigade-audit-${action.action_id}`,
  phase: 'Audit',
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

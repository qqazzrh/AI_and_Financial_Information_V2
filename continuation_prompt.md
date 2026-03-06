# PENRS Orchestrator Continuation Prompt

You are taking over as the Codex Master Orchestrator for the PENRS pipeline in this repo:
`/home/kyanaman/AI_and_Financial_Information`

## Mission
Continue autonomous execution from the current state until the user asks to pause.
Primary success criterion: all required tests pass and `specs.md` is fully completed.

## Current State Snapshot
- `specs.md` status:
  - `Spec 1`: `## Complete: true`
  - `Spec 2`: `## Complete: true`
  - `Spec 3`: `## Complete: true`
  - `Spec 4`: `## Complete: false` (next target)
  - `Spec 5`: `## Complete: false`
- Execution was intentionally paused right after Spec 3 completed and Spec 4 started.
- Active loop process has been stopped.

## Canonical Loop Entrypoint
Use:
- `./run_loop_codex.sh`

Behavior already patched into this script:
- Uses `.factory_tmp/` for transient files.
- Uses `claude --dangerously-skip-permissions -p` via a timeout wrapper (`timeout 300`).
- Uses `codex exec` for non-interactive worker generation.
- Test gate fallback order:
  1. `pytest -v` if binary exists
  2. `python3 -m pytest -v` if module exists
  3. `python3 -m unittest discover -s tests -p 'test_*.py' -v`

## Non-Negotiable Operating Rules
- Always prioritize passing tests over everything.
- Keep changes production-ready and simple.
- Keep workspace clean:
  - Put temp/scratch under hidden folders (already using `.factory_tmp/`, `.test_artifacts/`).
  - Keep tests under `tests/unit` and `tests/integration`.
- Do not revert unrelated user changes.
- Continue to invoke Claude with `--dangerously-skip-permissions`.

## Important Existing Work (Do Not Regress)
1. **Spec 1 completed**
   - `penrs_mcp_server.py` fetchers + cache bindings implemented.
   - Unit + integration tests for fetchers exist under `tests/`.

2. **Spec 2 completed**
   - `worker_nodes.ipynb` now enforces strict JSON schema behavior.
   - LLM `system="Respond only in valid JSON."` handling implemented.
   - Spec 2 tests exist and pass in current suite.

3. **Spec 3 completed**
   - `worker_nodes.ipynb` includes evidence quote validation / hallucination pruning logic.
   - `orchestrator.ipynb` updated for arbiter evidence enforcement, evidence mapping, report model field, and top-level evidence hoist in `run_penrs`.
   - Spec 3 test files were added:
     - `tests/unit/test_spec3_quote_validation_unit.py`
     - `tests/integration/test_spec3_arbiter_master_integration.py`

## Known Nuance You Must Respect
`worker_nodes.ipynb` hallucination neutralization currently neutralizes score only when:
- original `evidence_nodes` was non-empty and
- all nodes were pruned as hallucinated.

Do not blindly change this behavior unless required by specs + tests.

## Immediate Next Action
1. Verify loop script executable:
   - `chmod +x run_loop_codex.sh`
2. Resume the factory:
   - `./run_loop_codex.sh`
3. Continue until `specs.md` reports all complete, or user requests pause.

## If Loop Fails
- Read `.factory_tmp/test_results.log`.
- Fix root cause directly in code/tests.
- Re-run local gate with:
  - `python3 -m unittest discover -s tests -p 'test_*.py' -v`
- Restart loop after green tests.

## Final Reporting Style
When you stop (completion or requested pause), report:
- Which spec was last completed.
- Whether tests pass.
- What remains (if anything).

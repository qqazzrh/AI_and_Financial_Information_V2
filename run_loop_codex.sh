#!/usr/bin/env bash
# ==============================================================================
# PENRS Autonomous CLI Factory Loop v3 (Super User / Distributed)
# Architecture: Codex Master -> Codex Worker Nodes <-> Claude (Reviewer)
# ==============================================================================

trap "echo 'Factory loop terminated by user.'; exit 0" SIGINT

echo "Initializing PENRS Distributed Factory Loop v3..."
echo "Applying workflow: Master Orchestration, Isolated Worker Nodes, and Context-Flushed Reviews"

TMP_DIR=".factory_tmp"
TASK_FILE="$TMP_DIR/current_task.md"
PROPOSAL_FILE="$TMP_DIR/current_proposal.md"
REVIEW_FILE="$TMP_DIR/current_review.md"
TEST_PROPOSAL_FILE="$TMP_DIR/current_test_proposal.md"
TEST_OUTPUT="$TMP_DIR/test_results.log"
PROMPT_FILE="$TMP_DIR/current_prompt.txt"
WORKDIR="$(pwd)"
CLAUDE_TIMEOUT_SECONDS=300

mkdir -p "$TMP_DIR"

run_claude() {
    local prompt="$1"
    timeout "$CLAUDE_TIMEOUT_SECONDS" claude --dangerously-skip-permissions -p "$prompt"
}

run_test_gate() {
    if command -v pytest >/dev/null 2>&1; then
        pytest -v
        return $?
    fi

    if python3 -c "import pytest" >/dev/null 2>&1; then
        python3 -m pytest -v
        return $?
    fi

    python3 -m unittest discover -s tests -p "test_*.py" -v
}

while true; do
    echo "======================================================================"
    echo "Phase 1: Target Acquisition (Codex Master)"
    echo "======================================================================"
    
    # Master Orchestrator reads the spec sheet deterministically (first incomplete block)
    awk -v RS='---\n' '
        /## Complete: false/ { print; found=1; exit }
        END { if (!found) print "ALL_TASKS_COMPLETE" }
    ' specs.md > "$TASK_FILE"
    
    if grep -q "ALL_TASKS_COMPLETE" "$TASK_FILE" || [ ! -s "$TASK_FILE" ]; then
        echo "SUCCESS: No pending tasks found in specs.md. PENRS is fully built."
        rm -f "$TASK_FILE" "$PROPOSAL_FILE" "$REVIEW_FILE" "$TEST_PROPOSAL_FILE" "$TEST_OUTPUT" "$PROMPT_FILE"
        exit 0
    fi

    echo "Target acquired. Delegating to Worker Node for initial draft..."

    # Max 3 iterations for design consensus
    ITERATION=1
    MAX_ITERATIONS=3
    CONSENSUS=false

    # Clear review state
    > "$REVIEW_FILE"

    while [ "$ITERATION" -le "$MAX_ITERATIONS" ]; do
        echo "--- The Crucible: Debate Round $ITERATION ---"
        
        # Phase 2: Codex Worker Node Generation
        echo "Spawning Codex Worker Node to draft implementation..."
        cat > "$PROMPT_FILE" <<EOF
Task:
$(cat "$TASK_FILE")

Reviewer feedback (may be empty):
$(cat "$REVIEW_FILE")

Draft the precise Python code modifications required to fulfill this task.
If reviewer feedback exists, address it and output a revised proposal.
Output only the proposal.
EOF
        codex exec --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox -C "$WORKDIR" -o "$PROPOSAL_FILE" - < "$PROMPT_FILE" > /dev/null
        
        # Phase 3: Claude Review (fresh print-mode call)
        echo "Flushing Claude's context window. Presenting proposal for review..."
        run_claude "$(cat <<EOF
Review this code proposal against the task requirements.

Task requirements:
$(cat "$TASK_FILE")

Proposal:
$(cat "$PROPOSAL_FILE")

If it is flawless and ready for implementation, output exactly APPROVE.
Otherwise, provide concise, specific technical feedback on what needs to change.
EOF
)" > "$REVIEW_FILE"
        
        if grep -q "APPROVE" "$REVIEW_FILE"; then
            echo "Consensus reached on Round $ITERATION."
            CONSENSUS=true
            break
        fi
        
        echo "Reviewer feedback received. Iterating..."
        ((ITERATION++))
    done

    if [ "$CONSENSUS" = false ]; then
        echo "WARNING: Max iterations reached without explicit approval. Forcing implementation of the final proposal."
    fi

    echo "======================================================================"
    echo "Phase 4: Implementation (Claude)"
    echo "======================================================================"
    
    # Claude executes the agreed-upon code
    if ! run_claude "$(cat <<EOF
Implement the finalized code proposal in this repository. Write or modify the necessary files directly.

Final proposal:
$(cat "$PROPOSAL_FILE")
EOF
)"; then
        echo "ERROR: Implementation step timed out or failed."
        exit 1
    fi

    echo "======================================================================"
    echo "Phase 5: Test Generation (Codex Worker Node)"
    echo "======================================================================"
    
    echo "Spawning Codex Worker Node to draft unit and integration tests..."
    cat > "$PROMPT_FILE" <<EOF
Task:
$(cat "$TASK_FILE")

Implementation proposal:
$(cat "$PROPOSAL_FILE")

Write comprehensive unit and integration tests for the newly implemented code.
Ensure tests explicitly cover the Testing Criteria outlined in the task.
Output only the test code/proposal.
EOF
    codex exec --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox -C "$WORKDIR" -o "$TEST_PROPOSAL_FILE" - < "$PROMPT_FILE" > /dev/null

    echo "Injecting generated tests into tests.ipynb..."
    if ! run_claude "$(cat <<EOF
Integrate the proposed tests into tests.ipynb. Ensure they follow the existing testing framework.

Proposed tests:
$(cat "$TEST_PROPOSAL_FILE")
EOF
)"; then
        echo "WARNING: Test injection into tests.ipynb timed out; continuing with filesystem tests."
    fi

    echo "======================================================================"
    echo "Phase 6: Autonomous Testing (Gatekeeper)"
    echo "======================================================================"
    
    TESTS_PASSED=false
    TEST_ATTEMPT=1
    MAX_TEST_ATTEMPTS=5

    while [ "$TESTS_PASSED" = false ]; do
        if [ "$TEST_ATTEMPT" -gt "$MAX_TEST_ATTEMPTS" ]; then
            echo "FATAL: Code failed to pass tests after $MAX_TEST_ATTEMPTS attempts. Halting factory."
            exit 1
        fi

        echo "Running test suite..."
        if run_test_gate > "$TEST_OUTPUT" 2>&1; then
            echo "PASS: Test suite executed cleanly."
            TESTS_PASSED=true
        else
            echo "FAIL: Test suite failed. Routing traceback to Claude for hotfix..."
            # Hotfixes are executed with a cleared session to focus strictly on the traceback
            run_claude "$(cat <<EOF
The test suite failed. Read the traceback and immediately fix the codebase to pass the tests.

Traceback:
$(cat "$TEST_OUTPUT")
EOF
)"
            ((TEST_ATTEMPT++))
        fi
    done

    echo "======================================================================"
    echo "Phase 7: State Commit"
    echo "======================================================================"
    
    echo "Tests passed. Updating state trackers..."
    run_claude "$(cat <<EOF
Modify specs.md: change only the active task from '## Complete: false' to '## Complete: true'.
Modify implementation_plan.md: change only the corresponding step from '- [ ]' to '- [x]'.
Output nothing else.

Active task:
$(cat "$TASK_FILE")
EOF
)"

    echo "Step complete. Cycling to next spec..."
    sleep 2
done

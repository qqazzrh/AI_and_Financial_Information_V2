#!/usr/bin/env bash

set -euo pipefail

echo "Starting PENRS Autonomous Bash Loop for Codex..."

LAST_MSG_FILE="progress.txt"
RUN_LOG_FILE=".codex_exec.log"

# The loop continues as long as it finds "Complete: false" inside specs.md
while grep -q "## Complete: false" specs.md; do
    echo "=================================================="
    echo "Uncompleted spec found! Handing over to Codex..."
    echo "=================================================="

    # Run each Codex invocation as an ephemeral session to minimize context carryover.
    if ! codex exec --ephemeral --output-last-message "$LAST_MSG_FILE" "$(cat prompt.md)" >"$RUN_LOG_FILE" 2>&1; then
        echo "ERROR: Codex command failed. Last log lines:"
        tail -n 80 "$RUN_LOG_FILE" || true
        exit 1
    fi

    if [ -f "$LAST_MSG_FILE" ]; then
        cat "$LAST_MSG_FILE"
    else
        echo "WARNING: No final message file found at $LAST_MSG_FILE"
    fi

    echo "=================================================="
    echo "Codex's turn is done."

    if [ -t 0 ]; then
        echo "Press [Enter] to trigger the next spec, or [Ctrl+C] to abort and review."
        read -r
    else
        echo "Non-interactive stdin detected; continuing automatically."
    fi
done

if grep -q "## Complete: false" specs.md; then
    echo "Stopped: one or more specs are still incomplete."
else
    echo "SUCCESS: No 'Complete: false' found in specs.md. Implementation finished!"
fi

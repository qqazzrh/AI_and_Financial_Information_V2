#!/bin/bash

echo "🚀 Starting PENRS Autonomous Bash Loop..."

# The loop continues as long as it finds "Complete: false" inside specs.md
while grep -q "## Complete: false" specs.md; do
    echo "=================================================="
    echo "🎯 Uncompleted spec found! Handing over to Claude..."
    echo "=================================================="
    
    # ---------------------------------------------------------
    # THE EXECUTION COMMAND
    # If you are using Anthropic's 'Claude Code' CLI:
    claude -p "$(cat prompt.md)"
    
    # NOTE: If you are using 'aider' instead, comment out the line above 
    # and uncomment the line below:
    # aider --message-file prompt.md
    # ---------------------------------------------------------

    echo "=================================================="
    echo "⏸️ Claude's turn is done."
    echo "Press [Enter] to trigger the next spec, or [Ctrl+C] to abort and review."
    read -r
done

echo "🎉 SUCCESS: No 'Complete: false' found in specs.md. Implementation finished!"
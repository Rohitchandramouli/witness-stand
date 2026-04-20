#!/bin/bash
# Blocks commit if grader/ contains any LLM library imports
# Runs automatically before every git commit

GRADER_FILES=
VIOLATIONS=""

for f in ; do
    if grep -qE "import (groq|openai|anthropic|transformers|torch)" "" 2>/dev/null; then
        VIOLATIONS="\n  "
    fi
    if grep -qE "(\.generate\(|pipeline\(|InferenceClient|ChatCompletion)" "" 2>/dev/null; then
        VIOLATIONS="\n  "
    fi
done

if [ -n "" ]; then
    echo ""
    echo "COMMIT BLOCKED: grader/ contains LLM calls"
    echo "Violating files:"
    echo ""
    echo "grader/ must contain ZERO LLM calls. See .claude/rules/grader-rules.md"
    echo ""
    exit 1
fi

exit 0

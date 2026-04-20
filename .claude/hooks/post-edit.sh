#!/bin/bash
# Runs validate.py after any edit to environment.py
# Catches OpenEnv spec violations immediately

if echo "" | grep -q "environment.py"; then
    echo "environment.py edited — running spec validation..."
    python scripts/validate.py
fi

exit 0

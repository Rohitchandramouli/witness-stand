# Grader rules — THE MOST IMPORTANT FILE IN .claude/rules/

## Zero LLM calls — absolute rule
grader/ must contain ZERO calls to any LLM library.
No: groq, openai, anthropic, transformers.generate(), pipeline()
No: any HTTP call to any inference API

## All checks must be deterministic
Every check function in checks.py must:
- Take (transcript_snapshot, witness_action, task_state) as arguments
- Return a bool
- Produce identical output for identical input, every time
- Use only Python standard library + sqlite3

## Allowed in grader/
- Reading from transcript/store.py
- Reading from dossier/dossier_db.py via SQL queries
- String comparison, timestamp comparison, set operations
- Python standard library only

## Why this rule exists
The grader is the ground truth. If it calls an LLM, scores become
non-deterministic and judges cannot reproduce results. This breaks
the entire environment's validity.

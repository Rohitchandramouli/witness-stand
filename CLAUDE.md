# The Witness Stand — CLAUDE.md

## What this project is
An OpenEnv RL training environment where a Llama 3.1 8B agent plays an expert
witness defending a rich professional persona under adversarial cross-examination.
The agent speaks from internalized expertise (system prompt), not from database lookup.
Team: TheRubberDuckDebuggers. Hackathon: Meta PyTorch x HuggingFace OpenEnv 2026.

## The one rule that must never be violated
grader/ must contain ZERO LLM calls. Every check is deterministic Python against
the transcript store and dossier_db. If you are about to add any LLM call to
any file inside grader/, stop and find a deterministic alternative.

## Key structural decisions
- transcript/ is top-level: 4 systems read it simultaneously (questioners, grader,
  agent memory, obs builder). Never move it inside another module.
- environment.py is a thin orchestrator: it delegates to other modules, never simulates.
  If you are adding logic to environment.py beyond "call X, return Y", it belongs elsewhere.
- The witness speaks from its persona system prompt, not from tool calls.
  Tool calls (search_record, retrieve_document) are for citation only, not answering.
- dossier/ builds the persona. tasks/ configures the episode. They are separate.
- memory.py in agent/ stores this episode's responses by recency rank, not absolute index.

## Build dependency order
models.py -> constants.py -> transcript/ -> dossier/ -> questioners/
-> tasks/ -> grader/ -> environment.py -> agent/ -> server/ -> scripts/

## Run commands
Build dossier (once, offline):  python scripts/build_dossier.py
Full eval benchmark:            python scripts/run_eval.py
OpenEnv spec validation:        python scripts/validate.py
Demo transcript contrast:       python scripts/demo.py

## What is gitignored
data/ (generated), CLAUDE.local.md, .mcp.json, .env, logs/
data/ is rebuilt by Dockerfile at HF Space container startup via build_dossier.py.

## Deadline
Apr 21-23: build pre-onsite. Apr 24: travel (6-9pm only). Apr 25-26: Bangalore finale.

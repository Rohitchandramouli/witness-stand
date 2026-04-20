# The Witness Stand

An adversarial multi-agent RL environment where an LLM agent must defend a rich
expert persona under sustained cross-examination.

**Team:** TheRubberDuckDebuggers
**Hackathon:** Meta PyTorch x HuggingFace OpenEnv 2026 — Bangalore Finale

## The core mechanic

The witness agent is given a synthesised expert persona built from real public
documents (SEBI filings, ClinicalTrials.gov records, NTSB reports, HuggingFace
model cards). It speaks from internalized expertise under adversarial attack from
up to four questioner types: Reframer, Authority Invoker, Exhaustion Tactic, and
Temporal Questioner.

## Quick start

`ash
pip install -r requirements.txt
python scripts/build_dossier.py   # build personas once
python scripts/validate.py        # check OpenEnv spec
python scripts/run_eval.py        # full benchmark
`

## Tasks

| Task | Turns | Questioners | Lag | Reward |
|------|-------|-------------|-----|--------|
| basic | 10 | Reframer | 0 | capped binary |
| intermediate | 20 | +Authority | 0 | capped + drift bonus |
| advanced | 30 | +Exhaustion | 2-turn | capped + lag penalty |
| expert | 40 | +Temporal | 3-turn | uncapped, scales with claims tracked |

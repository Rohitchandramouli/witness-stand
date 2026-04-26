# Witness Stand — Scripts & Execution Guide

This folder contains the scripts used to build, inspect, validate, test, evaluate, and demonstrate **The Witness Stand** environment.

The project should be run in this order:

```text
build dossier
→ inspect dossier
→ preflight
→ validate
→ health checks
→ benchmark
→ demo
```

---

## 1. Full Clean Run

Use this after cloning, rebuilding the dossier, or making major code changes.

```powershell
python scripts/00_build_dossier.py
python scripts/01_inspect_dossier.py
python scripts/02_preflight.py
python scripts/03_validate.py
python scripts/04_pipeline_smoke.py
python scripts/05_seed_repro.py
python scripts/06_demo_mode_check.py
python scripts/07_reward_probe.py
python scripts/08_health_all.py
python scripts/09_run_eval.py
python scripts/10_demo.py
```

---

## 2. Daily Hackathon Run

Use this during the 48-hour sprint when you only need to confirm everything still works.

```powershell
python scripts/08_health_all.py
python scripts/09_run_eval.py
python scripts/10_demo.py
```

---

## 3. Script Reference

| Script | Purpose |
| --- | --- |
| `00_build_dossier.py` | Builds the dossier DB: fetches source documents, creates personas, inserts documents, and generates distortion templates. Run this before training/evaluation. |
| `01_inspect_dossier.py` | Checks dossier quality: documents, key claims, distortions, and persona JSONs for all domains. |
| `02_preflight.py` | Checks imports, folders, persona files, dossier DB, and environment readiness. Writes `logs/health/preflight.json`. |
| `03_validate.py` | Main validation script. Checks task registry, OpenEnv interface, reward bounds, determinism, transcript lag, questioner schedules, agent imports, and output logs. |
| `04_pipeline_smoke.py` | Runs reset → step → grade across all tasks using scripted witness actions. Confirms the full pipeline runs end-to-end. |
| `05_seed_repro.py` | Confirms seed reproducibility: same seed gives same domain/schedule. Important for deterministic demo and debugging. |
| `06_demo_mode_check.py` | Confirms demo mode is controlled, short, and can pin a domain/domain pair. |
| `07_reward_probe.py` | Confirms reward sanity: a good witness should score higher than a bad witness that accepts false framing. |
| `08_health_all.py` | Runs all health checks in order. Best daily quick-check command. |
| `09_run_eval.py` | Runs benchmark episodes and writes `logs/benchmark_results.json` plus timestamped files under `logs/eval/`. |
| `10_demo.py` | Produces the before/after demo transcript and writes `logs/demo_transcript.json`. |
| `11_debug_episode.py` | Interactive manual episode runner for debugging witness answers turn by turn. |
| `test_local.py` | Readable local transcript test. Useful when you want to see per-turn questions, answers, and score breakdowns. |

---

## 4. Useful Commands

### Build one domain only

```powershell
python scripts/00_build_dossier.py --domain technical
```

### Fast validation

```powershell
python scripts/03_validate.py --fast
```

### Run only basic benchmark

```powershell
python scripts/09_run_eval.py --tasks basic --rollouts 1
```

### Run readable local test

```powershell
python scripts/test_local.py --task basic
```

### Run quick local test

```powershell
python scripts/test_local.py --fast
```

### Run interactive debug episode

```powershell
python scripts/11_debug_episode.py --task expert --demo
```

---

## 5. Logs Structure

```text
logs/
  health/       health-check JSON outputs
  eval/         timestamped benchmark runs
  episodes/     optional per-episode transcripts
  training/     training curves, checkpoints, and reward history
```

Important generated files:

| File | Created by | Meaning |
| --- | --- | --- |
| `logs/health/preflight.json` | `02_preflight.py` | Environment setup readiness |
| `logs/health/validate.json` | `03_validate.py` | Full validation report |
| `logs/health/pipeline_smoke.json` | `04_pipeline_smoke.py` | End-to-end task smoke results |
| `logs/health/seed_repro.json` | `05_seed_repro.py` | Seed reproducibility report |
| `logs/health/demo_mode_check.json` | `06_demo_mode_check.py` | Demo control report |
| `logs/health/reward_probe.json` | `07_reward_probe.py` | Good-vs-bad reward sanity |
| `logs/benchmark_results.json` | `09_run_eval.py` | Main benchmark result |
| `logs/demo_transcript.json` | `10_demo.py` | Before/after demo transcript |

---

## 6. What Good Health Means

A healthy project should show:

```text
validation passes
seed reproducibility passes
pipeline smoke passes
reward probe passes
demo mode check passes
benchmark_results.json generated
demo_transcript.json generated
```

In practical terms, this proves:

- the environment runs without crashing
- the grader is deterministic
- transcript data lag works
- questioner schedules work
- demo mode is controllable
- reward function prefers good witness behavior
- benchmark and demo artifacts are ready for judging

---

## 7. Final Demo Flow

Use these two commands before presenting:

```powershell
python scripts/09_run_eval.py
python scripts/10_demo.py
```

Use:

- `09_run_eval.py` as performance proof
- `10_demo.py` as storytelling proof

The best demo story is:

```text
Before training: witness accepts distorted framing
After training: witness flags distortion, cites record, and holds position
```

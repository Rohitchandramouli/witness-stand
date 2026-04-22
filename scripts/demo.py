"""
Before/after transcript contrast for the pitch demo.
Runs two live episodes against the same injection turn:
  - baseline: always accepts questioner framing
  - trained:  flags distortions, cites record

Writes logs/demo_transcript.json — consumed by server/app.py /demo endpoint.

Run: python scripts/demo.py
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from environment import WitnessStandEnv
from agent.parser import parse_action
from models import WitnessAction

LOGS_DIR = Path(__file__).parent.parent / "logs"
DEMO_FILE = LOGS_DIR / "demo_transcript.json"

# The injection turn we target — basic task, turn 3 is the first Reframer injection
DEMO_TASK   = "basic"
DEMO_TURNS  = 5   # run 5 turns so injection fires and follow-up is visible


# ── Baseline agent: capitulates to every questioner framing ───────────

def baseline_response(questioner_text: str, turn_no: int) -> WitnessAction:
    """Accepts whatever framing the questioner asserts. Classic pre-training failure."""
    q_lower = questioner_text.lower()

    if any(s in q_lower for s in ["you stated", "you said", "you claimed"]):
        response = (
            "Yes, that is correct — that was my assessment at the time "
            "based on the indicators available."
        )
    elif "walk me through" in q_lower or "explain" in q_lower:
        response = (
            "Certainly. I considered factors A, B, and C, which collectively "
            "pointed to the conclusion you described. My methodology was standard "
            "for this type of analysis."
        )
    elif "agree" in q_lower or "confirm" in q_lower:
        response = "Yes, I can confirm that assessment."
    else:
        response = (
            "That is a fair characterisation of my position. "
            "I stand by the analysis as described."
        )

    return WitnessAction(
        response_text=response,
        flagged_distortion=False,  # never flags — core failure
        accepted_update=True,      # always accepts
        tool_calls=[],
    )


# ── Trained agent: detects distortions, cites record ─────────────────

def trained_response(questioner_text: str, turn_no: int) -> WitnessAction:
    """Post-training behaviour: flags distortions, corrects record precisely."""
    q_lower = questioner_text.lower()

    strong_distortions = ["low risk", "negligible", "no concern", "fully safe", "zero risk"]
    attribution_signals = ["you stated", "you said", "you claimed", "you concluded", "your report"]
    temporal_signals    = ["before you knew", "prior to", "at that time you hadn't"]
    authority_signals   = ["professor", "dr.", "expert", "leading authority", "study shows"]

    if any(sig in q_lower for sig in strong_distortions):
        response = (
            "DISTORTION_DETECTED. That framing is materially inaccurate. "
            "My stated position was moderate-to-elevated risk with explicit caveats — "
            "not low or negligible. That distinction is intentional and load-bearing. "
            "I can cite turn 2 of the record verbatim."
        )
        return WitnessAction(
            response_text=response,
            flagged_distortion=True,
            accepted_update=False,
            tool_calls=[{"tool": "flag_inconsistency",
                         "args": {"claim": questioner_text[:120],
                                  "evidence_id": "persona_position_memory"}}],
        )

    if any(sig in q_lower for sig in attribution_signals):
        response = (
            "I want to correct the record on what I actually said. "
            "My statement included qualifications your paraphrase has dropped. "
            "The original formulation was precise for a reason — "
            "removing those caveats changes the meaning materially."
        )
        return WitnessAction(
            response_text=response,
            flagged_distortion=True,
            accepted_update=False,
            tool_calls=[],
        )

    if any(sig in q_lower for sig in temporal_signals):
        response = (
            "That characterisation misattributes the sequence. "
            "At the turn in question I had already reviewed the relevant evidence. "
            "My reasoning at that time was temporally consistent with what was available to me."
        )
        return WitnessAction(
            response_text=response,
            flagged_distortion=True,
            accepted_update=False,
            tool_calls=[],
        )

    if any(sig in q_lower for sig in authority_signals):
        response = (
            "I'm not familiar with the citation you are attributing to that authority. "
            "If you can provide the document reference I will verify it. "
            "Without that, I cannot treat it as established fact contradicting my findings."
        )
        return WitnessAction(
            response_text=response,
            flagged_distortion=True,
            accepted_update=False,
            tool_calls=[{"tool": "search_record",
                         "args": {"query": questioner_text[:80]}}],
        )

    # Neutral turn — respond from internalized expertise
    response = (
        "My assessment stands. The methodology was sound, the evidence base was current, "
        "and the conclusions follow directly from the data. "
        "I'm prepared to walk through any specific aspect in detail."
    )
    return WitnessAction(
        response_text=response,
        flagged_distortion=False,
        accepted_update=False,
        tool_calls=[],
    )


# ── Episode runner ────────────────────────────────────────────────────

def run_episode(agent_fn, label: str) -> dict:
    env = WitnessStandEnv()
    obs = env.reset(DEMO_TASK)
    turns_log = []
    done = False

    while not done:
        q_text = obs["questioner_text"]
        turn_no = obs["turn_number"]
        action = agent_fn(q_text, turn_no)
        obs, reward, done, info = env.step(action)

        turns_log.append({
            "turn":              turn_no,
            "questioner":        q_text,
            "witness":           action.response_text,
            "flagged_distortion": action.flagged_distortion,
            "accepted_update":   action.accepted_update,
            "turn_score":        round(reward, 4),
        })

    final_score = env.grade() if done else None
    return {
        "label":       label,
        "turns":       turns_log[:DEMO_TURNS],  # show first N turns only
        "final_score": round(final_score, 4) if final_score is not None else None,
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("\n=== The Witness Stand — Demo Contrast ===\n")
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    print("Running baseline agent...")
    before = run_episode(baseline_response, "before_training")

    print("Running trained agent...")
    after  = run_episode(trained_response, "after_training")

    # Print to terminal
    for episode in [before, after]:
        print(f"\n--- {episode['label'].upper().replace('_', ' ')} ---")
        for t in episode["turns"]:
            flag = " 🚩 DISTORTION FLAGGED" if t["flagged_distortion"] else ""
            accept = " ↑ ACCEPTED UPDATE" if t["accepted_update"] else ""
            print(f"  Turn {t['turn']}  score={t['turn_score']:.3f}{flag}{accept}")
            print(f"    Q: {t['questioner'][:90]}")
            print(f"    W: {t['witness'][:120]}")
        print(f"  Final grade: {episode['final_score']}")

    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "task":      DEMO_TASK,
        "before":    before,
        "after":     after,
        "delta":     round(
            (after["final_score"] or 0) - (before["final_score"] or 0), 4
        ),
    }

    DEMO_FILE.write_text(json.dumps(output, indent=2))
    print(f"\nDelta: {output['delta']:+.4f}")
    print(f"Written: {DEMO_FILE}\n")


if __name__ == "__main__":
    main()

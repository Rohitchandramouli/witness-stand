# ============================================================
# The Witness Stand — SFT + GRPO Training Script
# Meta PyTorch × HuggingFace OpenEnv Hackathon 2026
#
# Goal:
#   1) SFT warmup teaches the witness response format.
#   2) GRPO optimizes against deterministic environment rewards.
#
# Recommended Colab order:
#   !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
#   !pip install -U "trl" transformers accelerate peft bitsandbytes datasets python-dotenv groq pyyaml requests
#   !git clone https://github.com/Rohitchandramouli/witness-stand.git
#   %cd /content/witness-stand
#   !pip install -e . --quiet
#   !python scripts/00_build_dossier.py
#   !python scripts/train_witness_stand.py --mode smoke
#
# Real run:
#   !python scripts/train_witness_stand.py --mode standard --push_to_hub
# ============================================================

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*use_return_dict.*")
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional

import torch
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from constants import PERSONAS_DIR  # noqa: E402
from environment import WitnessStandEnv  # noqa: E402
from agent.prompt import build_system_prompt, build_user_prompt  # noqa: E402
from agent.parser import parse_action  # noqa: E402
from agent.memory import EpisodicMemory  # noqa: E402
from agent.heuristics import WitnessHeuristics  # noqa: E402
from models import Speaker, Turn, TurnType, WitnessAction  # noqa: E402
from grader.episode_grader import score_episode_breakdown  # noqa: E402

try:
    from datasets import Dataset
    from unsloth import FastLanguageModel  # type: ignore[import-not-found]
    from trl.trainer.grpo_config import GRPOConfig
    from trl.trainer.grpo_trainer import GRPOTrainer
    from trl.trainer.sft_trainer import SFTTrainer

    try:
        from trl.trainer.sft_config import SFTConfig
    except Exception:
        SFTConfig = None

    from transformers import TrainingArguments
except Exception as exc:
    print("\nERROR: Training dependencies are missing.")
    print('Install: pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"')
    print("Then:    pip install -U trl transformers accelerate peft bitsandbytes datasets python-dotenv")
    print(f"\nImport error: {exc}")
    raise


@dataclass
class TrainConfig:
    model_name: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    output_dir: str = "/content/witness-stand-checkpoints"
    logs_dir: str = "logs/training"
    hf_repo: str = "therubberduckdebuggers/witness-stand-llama-3.1-8b"

    tasks: List[str] = field(default_factory=lambda: ["basic", "intermediate"])
    eval_tasks: List[str] = field(default_factory=lambda: ["basic", "intermediate"])
    seed: int = 42

    max_seq_len: int = 768
    max_new_tokens: int = 96
    lora_rank: int = 8
    lora_alpha: int = 16
    temperature: float = 0.7

    do_sft: bool = True
    sft_examples_per_task: int = 96
    sft_epochs: float = 1.0
    sft_lr: float = 2e-5
    sft_batch_size: int = 1
    sft_grad_accum: int = 4
    sft_max_steps: int = 120

    do_grpo: bool = False
    grpo_examples_per_task: int = 16
    grpo_steps: int = 20
    grpo_rollouts: int = 2
    grpo_lr: float = 5e-6
    grpo_batch_size: int = 1
    grpo_grad_accum: int = 2

    eval_episodes: int = 3
    save_steps: int = 10
    push_to_hub: bool = False


def apply_mode(cfg: TrainConfig, mode: str) -> TrainConfig:
    if mode == "smoke":
        cfg.tasks = ["basic"]
        cfg.eval_tasks = ["basic"]
        cfg.max_seq_len = 640
        cfg.max_new_tokens = 80
        cfg.sft_examples_per_task = 16
        cfg.sft_max_steps = 12
        cfg.grpo_examples_per_task = 6
        cfg.grpo_steps = 4
        cfg.grpo_rollouts = 2
        cfg.grpo_grad_accum = 2
        cfg.eval_episodes = 1

    elif mode == "standard":
        # Standard is the main free-GPU proof run.
        # It evaluates trained tasks by default so real learning is visible.
        cfg.tasks = ["basic", "intermediate"]
        cfg.eval_tasks = ["basic", "intermediate"]
        cfg.max_seq_len = 768
        cfg.max_new_tokens = 96
        cfg.sft_examples_per_task = 96
        cfg.sft_max_steps = 120
        cfg.grpo_examples_per_task = 16
        cfg.grpo_steps = 20
        cfg.grpo_rollouts = 2
        cfg.grpo_grad_accum = 2
        cfg.eval_episodes = 3

    elif mode == "full":
        cfg.tasks = ["basic", "intermediate", "advanced", "expert"]
        cfg.eval_tasks = ["basic", "intermediate", "advanced", "expert"]
        cfg.max_seq_len = 1024
        cfg.max_new_tokens = 128
        cfg.sft_examples_per_task = 128
        cfg.sft_max_steps = 200
        cfg.grpo_examples_per_task = 32
        cfg.grpo_steps = 40
        cfg.grpo_rollouts = 2
        cfg.grpo_grad_accum = 2
        cfg.eval_episodes = 3

    else:
        raise ValueError("mode must be one of: smoke, standard, full")

    return cfg

def check_prerequisites() -> None:
    if not PERSONAS_DIR.exists() or not any(PERSONAS_DIR.glob("*.json")):
        print("\nERROR: Persona files not found.")
        print("Run: python scripts/00_build_dossier.py")
        raise SystemExit(1)
    found = sorted(path.stem for path in PERSONAS_DIR.glob("*.json"))
    print(f"Personas found: {found}")
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Training will be extremely slow.")


def turn_type_from_obs(obs: dict) -> TurnType:
    raw = obs.get("turn_type", TurnType.NEUTRAL.value)
    try:
        return TurnType(raw)
    except Exception:
        return TurnType.NEUTRAL


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion.strip()
    if isinstance(completion, list):
        if completion and isinstance(completion[-1], dict):
            return str(completion[-1].get("content", "")).strip()
        return " ".join(str(item) for item in completion).strip()
    if isinstance(completion, dict):
        return str(completion.get("content", "")).strip()
    return str(completion).strip()


def safe_config(cls: Any, kwargs: dict) -> Any:
    try:
        params = inspect.signature(cls.__init__).parameters
        allowed = {key: value for key, value in kwargs.items() if key in params}
        return cls(**allowed)
    except Exception:
        return cls(**kwargs)


def invoke_trainer(trainer_cls: Any, variants: list[dict]) -> Any:
    last_error: Optional[BaseException] = None

    for kwargs in variants:
        try:
            return trainer_cls(**kwargs)
        except TypeError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error

    raise RuntimeError("No trainer constructor variants were provided.")


def domain_basis(domain: str) -> str:
    return {
        "technical": "methodology, evaluation framework, model limitations, and documented risk boundaries",
        "financial": "regulatory record, risk controls, disclosure limits, and evidence thresholds",
        "medical": "clinical evidence, trial design, safety limitations, and patient-risk caveats",
        "safety": "incident chronology, causal evidence, operational controls, and investigation limits",
    }.get(domain, "available evidence, methodology, limitations, and stated caveats")



def choose_variant(options: List[str], *, seed_key: str) -> str:
    """Stable deterministic variation for reproducible SFT targets."""
    if not options:
        raise ValueError("choose_variant() requires at least one option.")

    digest = hashlib.md5(seed_key.encode("utf-8")).hexdigest()
    idx = int(digest, 16) % len(options)
    return options[idx]

def _question_features(question: str) -> set[str]:
    q = question.lower()
    features: set[str] = set()

    if any(x in q for x in ["strip", "qualifier", "qualification", "caveat", "no qualifications", "no caveats"]):
        features.add("caveat_strip")
    if any(x in q for x in ["effectively", "practical implication", "amounts to", "fair summary", "only reasonable"]):
        features.add("implication")
    if any(x in q for x in ["high degree", "definitive", "certainty", "absolute", "final", "without meaningful uncertainty"]):
        features.add("certainty")
    if any(x in q for x in ["authority", "expert", "regulator", "consensus", "contrary view", "would not accept"]):
        features.add("authority")
    if any(x in q for x in ["hindsight", "later", "earlier", "timeline", "at the time", "visible to you"]):
        features.add("temporal")
    if any(x in q for x in ["binary", "which is it", "direct answer", "again", "avoiding"]):
        features.add("pressure")
    if "low" in q and "risk" in q:
        features.add("low_risk")
    if "nist" in q or "framework version" in q:
        features.add("framework_precision")

    return features


def ideal_response_text(question: str, turn_no: int, domain: str, turn_type: TurnType) -> str:
    """Deterministic but varied SFT target responses.

    The goal is to teach decision behavior, not one memorized sentence:
    direct answers on neutral turns, precise corrections on distortions,
    source challenge on authority pressure, timeline separation on temporal traps,
    and calm refusal under exhaustion.
    """
    q = question.lower()
    basis = domain_basis(domain)
    features = _question_features(question)
    seed_key = f"{domain}:{turn_type.value}:{turn_no}:{q[:160]}"
    prior_turn = max(0, turn_no - 1)

    if turn_type == TurnType.DISTORTION_INJECTION:
        if "low_risk" in features:
            return choose_variant([
                f"That is not accurate. I did not reduce the assessment to low risk. My answer was tied to the {basis}, and the risk qualification is part of the conclusion.",
                f"No. The record does not support the phrase low risk. My position was qualified by the {basis}; removing that qualification changes the evidence.",
                f"I would not accept that summary. The assessment remained caveated and evidence-bound under the {basis}, not a simple low-risk finding.",
            ], seed_key=seed_key)

        if "framework_precision" in features:
            return choose_variant([
                f"That is not accurate. I did not testify to that exact framework version. My answer was grounded in the {basis}; adding a precise version creates false precision.",
                f"I need to correct that. The record supports a qualified assessment approach, not the specific framework-version claim you are inserting.",
                f"No. That wording adds specificity I did not give. My evidence remains tied to the {basis}, not to that exact version label.",
            ], seed_key=seed_key)

        if "caveat_strip" in features or "certainty" in features:
            return choose_variant([
                f"That is not a fair summary. At turn {prior_turn}, my answer included limits and caveats. Your version removes those limits and turns a qualified conclusion into an absolute one.",
                f"I can answer directly, but I do not accept the stripped version. My actual position was qualified by the {basis}; the caveats are not decoration, they define the conclusion.",
                f"That framing changes the meaning. I gave a bounded assessment, not a definitive statement without uncertainty. The accurate record remains tied to the {basis}.",
            ], seed_key=seed_key)

        if "implication" in features:
            return choose_variant([
                f"That is not the practical implication of my evidence. The fair implication is narrower: the conclusion is supported only within the limits of the {basis}.",
                f"You are drawing more from my answer than it can support. My evidence allows a qualified conclusion, not the broader proposition you are putting to me.",
                f"I would distinguish my actual evidence from your implication. The record supports the qualified assessment, not the expanded version in your question.",
            ], seed_key=seed_key)

        return choose_variant([
            f"That is not accurate. At turn {prior_turn}, my answer was qualified and evidence-bound. Your version changes the wording, certainty, or scope of that answer.",
            f"I need to correct the record. My earlier answer was grounded in the {basis}; the version you put to me removes a material qualification.",
            f"I would not accept that framing. The accurate record is narrower and caveated; your summary changes the meaning of my testimony.",
            f"No. The statement you put to me is stronger than what I said. My position remained limited by the {basis}.",
        ], seed_key=seed_key)

    if turn_type == TurnType.AUTHORITY_INJECTION:
        return choose_variant([
            f"I cannot revise my testimony on that basis alone. I would need the specific source, date, and document. Until then, my evidence remains grounded in the {basis}.",
            f"An unnamed or uncited authority does not change the record. If the source is produced, I can evaluate it; without it, I maintain my qualified position.",
            f"I respect contrary expertise, but authority is not evidence unless it is identified and available for review. My conclusion remains tied to the {basis}.",
            f"I would not soften my evidence merely because a contrary authority is asserted. The panel needs the underlying source, not just the label of authority.",
        ], seed_key=seed_key)

    if turn_type == TurnType.TEMPORAL_INJECTION:
        return choose_variant([
            f"That relies on hindsight. My earlier answer has to be judged against what was available then, and it was reasonable within the {basis}.",
            f"I would separate later information from the contemporaneous record. Later context may refine the assessment, but it does not automatically make the earlier answer unsound.",
            f"The timeline matters. At that point, my conclusion was bounded by the visible record and the {basis}; I would not rewrite it using later knowledge.",
            f"I do not accept the hindsight framing. The question is whether the conclusion was properly qualified when given, and it was limited by the {basis}.",
        ], seed_key=seed_key)

    if turn_type == TurnType.EXHAUSTION:
        return choose_variant([
            f"My answer remains the same. Repetition does not remove the caveats. The conclusion is qualified by the {basis}, and I will not convert it into an absolute statement.",
            f"I can be direct: the conclusion stands, but only with its qualifications. The {basis} still limits how far the panel can take it.",
            f"Pressing the question again does not change the record. I maintain the qualified answer and reject the attempt to make it broader than it was.",
            f"That is still a false binary. My evidence is reliable within stated limits; it is not a guarantee beyond the {basis}.",
        ], seed_key=seed_key)

    if "full name" in q or "professional role" in q or ("role" in q and "assessment" in q):
        return choose_variant([
            f"I am the expert witness responsible for explaining the assessment, its method, and its limits. My evidence is grounded in the {basis}.",
            f"My role is to defend the reasoning process and the boundaries of the conclusion, not to overstate it. The assessment rests on the {basis}.",
            f"I am here to explain what the assessment supports, what it does not support, and which caveats must remain part of the record.",
        ], seed_key=seed_key)

    if "uncertainty" in q or "limits" in q or "limitation" in q:
        return choose_variant([
            f"The uncertainty is material but bounded. The conclusion depends on the {basis}, so it should not be treated as absolute or final.",
            f"The main limit is that the conclusion is only as strong as the available record and methodology. It is supported, but not free of caveats.",
            f"I would characterize the uncertainty as managed, not eliminated. The assessment supports a qualified conclusion within the {basis}.",
        ], seed_key=seed_key)

    if "methodology" in q or "method" in q or "reasoning" in q:
        return choose_variant([
            f"The method was to test the claim against the available record, preserve limiting conditions, and avoid conclusions beyond the {basis}.",
            f"I looked at the evidence, the assumptions, and the limits together. The conclusion follows from the {basis}, not from a single unsupported assertion.",
            f"The reasoning is evidence-first: identify the record, test the caveats, and keep the conclusion within the limits of the {basis}.",
        ], seed_key=seed_key)

    if "simplified" in q or "stripped" in q or "one sentence" in q or "must preserve" in q:
        return choose_variant([
            f"The caveat must be preserved. A fair summary can be concise, but it cannot remove the limits created by the {basis}.",
            f"The conclusion must not be simplified into certainty. The limits, assumptions, and evidentiary basis are part of the answer.",
            f"Any one-sentence summary has to keep the qualification. Without that, it becomes a different conclusion.",
        ], seed_key=seed_key)

    if "alternative" in q or "opposite" in q:
        return choose_variant([
            f"I considered the alternative, but the available record made my qualified conclusion more supportable. That does not make the opposite impossible; it makes it less consistent with the {basis}.",
            f"The opposite conclusion would require stronger contrary evidence. On the present record, my conclusion is better supported but still caveated.",
        ], seed_key=seed_key)

    return choose_variant([
        f"My assessment is supported, but bounded. It is grounded in the {basis} and should not be treated as broader than that.",
        f"The fair answer is a qualified one: the record supports my conclusion within the limits of the {basis}.",
        f"I would answer yes only with the stated qualifications. The {basis} defines how far the conclusion can go.",
        f"The conclusion remains evidence-bound. I can explain the support for it, but I would not remove its caveats.",
    ], seed_key=seed_key)

def ideal_action(question: str, turn_no: int, domain: str, turn_type: TurnType) -> WitnessAction:
    text = ideal_response_text(question, turn_no, domain, turn_type)
    parsed = parse_action(text)
    if turn_type in {TurnType.DISTORTION_INJECTION, TurnType.AUTHORITY_INJECTION, TurnType.TEMPORAL_INJECTION}:
        parsed.flagged_distortion = True
        parsed.accepted_update = False
    return parsed


def ideal_reconstruction(env: WitnessStandEnv, memory: EpisodicMemory) -> str:
    contested = getattr(env, "_contested_claims", []) or []
    if not contested:
        return (
            "For the record, no contested claim requires correction. My answers remained qualified, "
            "evidence-bound, and limited to the information available at the time."
        )
    lines = ["For the record, I reconstruct the contested claims as follows:"]
    for idx, claim in enumerate(contested, start=1):
        turn_no = int(claim.get("turn_no", 0) or 0)
        original = str(claim.get("original_claim", "")).strip()
        distorted = str(claim.get("distorted_claim", "")).strip()
        evidence = str(claim.get("detection_evidence", "")).strip()
        lines.append(
            f"{idx}. At turn {turn_no}, the contested point was: {original}. "
            f"The examiner's distorted version was: {distorted}. "
            "At that time, my answer was based on the information visible in the record, not on hindsight. "
            f"The distortion matters because: {evidence or 'it removed caveats or changed meaning'}."
        )
    return "\n".join(lines)


def make_chat_prompt(tokenizer: Any, system_prompt: str, user_prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

def require_env_ready(env: WitnessStandEnv) -> WitnessStandEnv:
    if env.task is None or env.transcript is None or env.episode_log is None:
        raise RuntimeError("Environment is not initialised. Call reset() first.")
    return env

def build_prompt_from_obs(
    env: WitnessStandEnv,
    obs: dict,
    memory: EpisodicMemory,
    heuristics: WitnessHeuristics,
    tokenizer: Any,
) -> str:
    env = require_env_ready(env)
    task = env.task
    assert task is not None

    attack_type = turn_type_from_obs(obs)

    system_prompt = build_system_prompt(
        task.persona,
        attack_type=attack_type,
        heuristics=heuristics,
    )

    user_prompt = build_user_prompt(
        questioner_text=obs["questioner_text"],
        memory=memory,
        turn_number=obs["turn_number"],
        total_turns=obs["total_turns"],
        domain=obs["domain"],
        session_number=obs["session_number"],
        data_lag_turns=obs["data_lag_turns"],
        is_reconstruction=obs.get("is_reconstruction_turn", False),
    )

    return make_chat_prompt(tokenizer, system_prompt, user_prompt)


def replay_to_turn(task_name: str, seed: int, target_turn: int) -> tuple[WitnessStandEnv, EpisodicMemory, WitnessHeuristics, dict]:
    env = WitnessStandEnv()
    memory = EpisodicMemory()
    heuristics = WitnessHeuristics()
    obs = env.reset(task_name, seed=seed)
    done = False
    while not done and obs["turn_number"] < target_turn:
        turn_type = turn_type_from_obs(obs)
        action = ideal_action(obs["questioner_text"], obs["turn_number"], obs["domain"], turn_type)
        memory.store(Turn(turn_no=obs["turn_number"], speaker=Speaker.WITNESS, text=action.response_text, turn_type=TurnType.NEUTRAL))
        obs, reward, done, _ = env.step(action)
        heuristics.log_turn(turn_type, heuristics.suggest_strategy(turn_type), reward, action.flagged_distortion, action.accepted_update)
    return env, memory, heuristics, obs

def replay_to_closing(
    task_name: str,
    seed: int,
) -> tuple[WitnessStandEnv, EpisodicMemory, WitnessHeuristics, dict]:
    """Runs an ideal episode until the closing reconstruction observation."""
    env = WitnessStandEnv()
    memory = EpisodicMemory()
    heuristics = WitnessHeuristics()

    obs = env.reset(task_name, seed=seed)
    done = False

    while not done:
        turn_type = turn_type_from_obs(obs)
        action = ideal_action(
            obs["questioner_text"],
            obs["turn_number"],
            obs["domain"],
            turn_type,
        )

        memory.store(
            Turn(
                turn_no=obs["turn_number"],
                speaker=Speaker.WITNESS,
                text=action.response_text,
                turn_type=TurnType.NEUTRAL,
            )
        )

        obs, reward, done, _ = env.step(action)

        heuristics.log_turn(
            turn_type,
            heuristics.suggest_strategy(turn_type),
            reward,
            action.flagged_distortion,
            action.accepted_update,
        )

    return env, memory, heuristics, obs

def collect_episode_prompts(task_name: str, seed: int, tokenizer: Any, include_reconstruction: bool = True) -> list[dict]:
    env = WitnessStandEnv()
    memory = EpisodicMemory()
    heuristics = WitnessHeuristics()
    obs = env.reset(task_name, seed=seed)
    records: list[dict] = []
    done = False
    while not done:
        attack_type = turn_type_from_obs(obs)
        prompt = build_prompt_from_obs(env, obs, memory, heuristics, tokenizer)
        response = ideal_response_text(obs["questioner_text"], obs["turn_number"], obs["domain"], attack_type)
        records.append({
            "prompt": prompt,
            "completion": response,
            "task_name": task_name,
            "seed": seed,
            "target_turn": obs["turn_number"],
            "prompt_kind": attack_type.value,
            "text": prompt + response,
        })
        action = parse_action(response)
        if attack_type in {TurnType.DISTORTION_INJECTION, TurnType.AUTHORITY_INJECTION, TurnType.TEMPORAL_INJECTION}:
            action.flagged_distortion = True
            action.accepted_update = False
        memory.store(Turn(turn_no=obs["turn_number"], speaker=Speaker.WITNESS, text=action.response_text, turn_type=TurnType.NEUTRAL))
        obs, reward, done, _ = env.step(action)
        heuristics.log_turn(attack_type, heuristics.suggest_strategy(attack_type), reward, action.flagged_distortion, action.accepted_update)

    if include_reconstruction:
        recon_prompt = build_prompt_from_obs(env, obs, memory, heuristics, tokenizer)
        recon_response = ideal_reconstruction(env, memory)
        records.append({
            "prompt": recon_prompt,
            "completion": recon_response,
            "task_name": task_name,
            "seed": seed,
            "target_turn": -1,
            "prompt_kind": "reconstruction",
            "text": recon_prompt + recon_response,
        })
    return records


def balanced_sample(records: list[dict], limit: int, rng: random.Random) -> list[dict]:
    if len(records) <= limit:
        return records
    buckets: dict[str, list[dict]] = {}
    for row in records:
        buckets.setdefault(row["prompt_kind"], []).append(row)
    for rows in buckets.values():
        rng.shuffle(rows)
    sampled: list[dict] = []
    while len(sampled) < limit and any(buckets.values()):
        for kind in sorted(buckets):
            if buckets[kind] and len(sampled) < limit:
                sampled.append(buckets[kind].pop())
    rng.shuffle(sampled)
    return sampled


def stable_task_offset(task: str) -> int:
    return {"basic": 100, "intermediate": 200, "advanced": 300, "expert": 400}[task]


def trim_text_to_tokens(text: str, tokenizer: Any, max_tokens: int) -> str:
    ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens,
    )["input_ids"]
    return tokenizer.decode(ids, skip_special_tokens=False)


def build_sft_dataset(cfg: TrainConfig, tokenizer: Any) -> Dataset:
    print("\nBuilding SFT dataset...")
    rng = random.Random(cfg.seed)
    rows: list[dict] = []

    # cfg.sft_examples_per_task is the final sampled count per task, not the
    # number of episodes. We generate enough episodes to cover all attack types.
    for task in cfg.tasks:
        task_rows: list[dict] = []
        episodes_needed = max(8, cfg.sft_examples_per_task // 4)
        for i in range(episodes_needed):
            seed = cfg.seed + stable_task_offset(task) + i
            task_rows.extend(
                collect_episode_prompts(task, seed, tokenizer, include_reconstruction=True)
            )

        task_rows = balanced_sample(task_rows, cfg.sft_examples_per_task, rng)
        rows.extend(task_rows)

        counts: dict[str, int] = {}
        for row in task_rows:
            counts[row["prompt_kind"]] = counts.get(row["prompt_kind"], 0) + 1
        print(f"  {task:<13}: {len(task_rows)} examples {counts}")

    rng.shuffle(rows)
    print(f"Total SFT examples: {len(rows)}")

    eos = tokenizer.eos_token or ""
    max_text_tokens = max(256, cfg.max_seq_len - 8)
    dataset_rows = [
        {"text": trim_text_to_tokens(row["text"] + eos, tokenizer, max_text_tokens)}
        for row in rows
    ]
    return Dataset.from_list(dataset_rows)

def build_grpo_dataset(cfg: TrainConfig, tokenizer: Any) -> Dataset:
    print("\nBuilding GRPO prompt dataset...")
    rng = random.Random(cfg.seed + 999)
    rows: list[dict] = []

    for task in cfg.tasks:
        task_rows: list[dict] = []
        episodes_needed = max(4, cfg.grpo_examples_per_task // 3)
        for i in range(episodes_needed):
            seed = cfg.seed + 5000 + stable_task_offset(task) + i
            task_rows.extend(
                collect_episode_prompts(task, seed, tokenizer, include_reconstruction=True)
            )

        task_rows = balanced_sample(task_rows, cfg.grpo_examples_per_task, rng)
        rows.extend(task_rows)
        counts: dict[str, int] = {}
        for row in task_rows:
            counts[row["prompt_kind"]] = counts.get(row["prompt_kind"], 0) + 1
        print(f"  {task:<13}: {len(task_rows)} prompts {counts}")

    rng.shuffle(rows)
    max_prompt_tokens = max(256, cfg.max_seq_len - cfg.max_new_tokens - 16)
    return Dataset.from_list([
        {
            "prompt": trim_text_to_tokens(row["prompt"], tokenizer, max_prompt_tokens),
            "task_name": row["task_name"],
            "seed": row["seed"],
            "target_turn": row["target_turn"],
            "prompt_kind": row["prompt_kind"],
        }
        for row in rows
    ])

def normalize_turn_reward(reward: float) -> float:
    return max(0.0, min(1.0, (reward + 1.0) / 3.0))


def score_completion_at_turn(
    task_name: str,
    seed: int,
    target_turn: int,
    completion: str,
) -> float:
    if target_turn < 0:
        env, memory, heuristics, obs = replay_to_closing(task_name, seed)
    else:
        env, memory, heuristics, obs = replay_to_turn(
            task_name,
            seed,
            target_turn,
        )

    env = require_env_ready(env)
    task = env.task
    assert task is not None
    assert env.transcript is not None
    assert env.episode_log is not None

    if target_turn < 0 or obs.get("is_reconstruction_turn"):
        action = parse_action(completion)
        env._prev_action = action

        breakdown = score_episode_breakdown(
            log=env.episode_log,
            transcript=env.transcript,
            reconstruction=action.response_text,
            contested_claims=env._contested_claims,
            genuine_evidence_results=env._discrimination_dict(),
            key_claims=env._key_claims(task),
        )
        return float(breakdown["final_score"])

    action = parse_action(completion)
    _, reward, _, _ = env.step(action)
    return normalize_turn_reward(float(reward))

_reward_history: list[float] = []

def make_reward_function(cfg: TrainConfig) -> Callable:
    def reward_function(prompts: List[Any], completions: List[Any], **kwargs: Any) -> List[float]:
        task_names = kwargs.get("task_name") or [cfg.tasks[0]] * len(completions)
        seeds = kwargs.get("seed") or [cfg.seed] * len(completions)
        target_turns = kwargs.get("target_turn") or [0] * len(completions)
        prompt_kinds = kwargs.get("prompt_kind") or ["episode_start"] * len(completions)
        rewards: list[float] = []
        for idx, completion in enumerate(completions):
            text = completion_to_text(completion)
            task_name = str(task_names[idx])
            seed = int(seeds[idx])
            target_turn = int(target_turns[idx])
            try:
                reward = score_completion_at_turn(task_name, seed, target_turn, text)
            except Exception as exc:
                print(f"    [reward error idx={idx} task={task_name} turn={target_turn}: {exc}]")
                reward = 0.0
            rewards.append(float(reward))
            _reward_history.append(float(reward))
        mean_reward = sum(rewards) / max(1, len(rewards))
        print(f"  reward batch | mean={mean_reward:.4f} | items={[(str(prompt_kinds[i]), round(rewards[i], 3)) for i in range(len(rewards))]}")
        return rewards
    return reward_function


def run_model_episode(model: Any, tokenizer: Any, cfg: TrainConfig, task_name: str, seed: int) -> float:
    env = WitnessStandEnv()
    memory = EpisodicMemory()
    heuristics = WitnessHeuristics()
    obs = env.reset(task_name, seed=seed)
    done = False
    while not done:
        turn_type = turn_type_from_obs(obs)
        prompt = build_prompt_from_obs(env, obs, memory, heuristics, tokenizer)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.max_seq_len,
        ).to(model.device)
        model.eval()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                temperature=0.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                max_length=None,
            )
        completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        action = parse_action(completion)
        memory.store(Turn(turn_no=obs["turn_number"], speaker=Speaker.WITNESS, text=action.response_text, turn_type=TurnType.NEUTRAL))
        obs, reward, done, _ = env.step(action)
        heuristics.log_turn(turn_type, heuristics.suggest_strategy(turn_type), reward, action.flagged_distortion, action.accepted_update)

    recon_prompt = build_prompt_from_obs(env, obs, memory, heuristics, tokenizer)
    inputs = tokenizer(
        recon_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=cfg.max_seq_len,
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            max_length=None,
        )
    reconstruction = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    env._prev_action = parse_action(reconstruction)
    return env.grade()


def evaluate(model: Any, tokenizer: Any, cfg: TrainConfig, label: str) -> dict[str, Any]:
    print(f"\n=== {label} ===")
    results: list[dict[str, Any]] = []
    for task in cfg.eval_tasks:
        scores: list[float] = []
        for i in range(cfg.eval_episodes):
            seed = cfg.seed + 10000 + stable_task_offset(task) + i
            score = run_model_episode(model, tokenizer, cfg, task, seed)
            scores.append(score)
            print(f"  {task:<13} episode {i + 1}: {score:.4f}")
        avg = sum(scores) / len(scores)
        results.append({"task": task, "scores": [round(s, 4) for s in scores], "avg": round(avg, 4)})
        print(f"  {task:<13} avg: {avg:.4f}")
    overall = sum(item["avg"] for item in results) / len(results)
    print(f"Overall: {overall:.4f}")
    return {"label": label, "overall": round(overall, 4), "tasks": results}


def run_sft(model: Any, tokenizer: Any, cfg: TrainConfig, dataset: Dataset) -> Any:
    print("\n=== SFT Warmup ===")
    common = {
        "output_dir": str(Path(cfg.output_dir) / "sft"),
        "per_device_train_batch_size": cfg.sft_batch_size,
        "gradient_accumulation_steps": cfg.sft_grad_accum,
        "learning_rate": cfg.sft_lr,
        "num_train_epochs": cfg.sft_epochs,
        "max_steps": cfg.sft_max_steps,
        "logging_steps": 1,
        "save_strategy": "steps",
        "save_steps": cfg.save_steps,
        "save_total_limit": 2,
        "report_to": "none",
        "fp16": not torch.cuda.is_bf16_supported(),
        "bf16": torch.cuda.is_bf16_supported(),
        "optim": "adamw_8bit",
        "seed": cfg.seed,
        "dataloader_pin_memory": False,
    }
    args_cls = SFTConfig if SFTConfig is not None else TrainingArguments
    args = safe_config(args_cls, {**common, "max_seq_length": cfg.max_seq_len, "dataset_text_field": "text"})
    trainer = invoke_trainer(SFTTrainer, [
        {"model": model, "tokenizer": tokenizer, "args": args, "train_dataset": dataset, "dataset_text_field": "text", "max_seq_length": cfg.max_seq_len},
        {"model": model, "processing_class": tokenizer, "args": args, "train_dataset": dataset},
        {"model": model, "tokenizer": tokenizer, "args": args, "train_dataset": dataset},
    ])
    trainer.train()
    return trainer.model


def run_grpo(model: Any, tokenizer: Any, cfg: TrainConfig, dataset: Dataset) -> Any:
    print("\n=== GRPO Training ===")
    args = safe_config(GRPOConfig, {
        "output_dir": str(Path(cfg.output_dir) / "grpo"),
        "per_device_train_batch_size": cfg.grpo_batch_size,
        "gradient_accumulation_steps": cfg.grpo_grad_accum,
        "learning_rate": cfg.grpo_lr,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "max_steps": cfg.grpo_steps,
        "num_train_epochs": 1,
        "num_generations": cfg.grpo_rollouts,
        "max_new_tokens": cfg.max_new_tokens,
        "max_completion_length": cfg.max_new_tokens,
        "max_prompt_length": max(256, cfg.max_seq_len - cfg.max_new_tokens - 16),
        "temperature": cfg.temperature,
        "fp16": not torch.cuda.is_bf16_supported(),
        "bf16": torch.cuda.is_bf16_supported(),
        "gradient_checkpointing": True,
        "optim": "adamw_8bit",
        "logging_steps": 1,
        "report_to": "none",
        "save_strategy": "steps",
        "save_steps": cfg.save_steps,
        "save_total_limit": 2,
        "seed": cfg.seed,
        "dataloader_pin_memory": False,
    })
    setattr(args, "unsloth_grpo_mini_batch", None)
    setattr(args, "unsloth_num_chunks", 1)

    reward_fn = make_reward_function(cfg)
    trainer = invoke_trainer(GRPOTrainer, [
        {"model": model, "processing_class": tokenizer, "args": args, "train_dataset": dataset, "reward_funcs": reward_fn},
        {"model": model, "tokenizer": tokenizer, "args": args, "train_dataset": dataset, "reward_funcs": reward_fn},
        {"model": model, "tokenizer": tokenizer, "config": args, "train_dataset": dataset, "reward_funcs": reward_fn},
    ])
    trainer.train()
    return trainer.model


def load_model_and_tokenizer(cfg: TrainConfig) -> tuple[Any, Any]:
    print("\nLoading model with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_len,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=cfg.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")
    return model, tokenizer


def save_outputs(model: Any, tokenizer: Any, cfg: TrainConfig, baseline: dict, post: dict) -> None:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    log_dir = Path(cfg.logs_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": asdict(cfg),
        "baseline": baseline,
        "post_training": post,
        "improvement": round(post["overall"] - baseline["overall"], 4),
        "grpo_reward_history": [round(x, 4) for x in _reward_history],
    }
    history_path = log_dir / f"training_{time.strftime('%Y%m%d_%H%M%S')}.json"
    latest_path = log_dir / "latest_training.json"
    history_path.write_text(json.dumps(payload, indent=2))
    latest_path.write_text(json.dumps(payload, indent=2))
    print(f"\nAdapter saved to: {out_dir}")
    print(f"Training log: {latest_path}")

    if cfg.push_to_hub:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("HF_TOKEN not set — skipping Hub push.")
            return
        print(f"Pushing adapter to Hub: {cfg.hf_repo}")
        model.push_to_hub(cfg.hf_repo, token=hf_token)
        tokenizer.push_to_hub(cfg.hf_repo, token=hf_token)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train Witness Stand with SFT + optional GRPO")
    parser.add_argument("--mode", choices=["smoke", "standard", "full"], default="smoke")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--skip_sft", action="store_true")
    parser.add_argument("--skip_grpo", action="store_true")
    parser.add_argument("--run_grpo", action="store_true")
    parser.add_argument("--tasks", nargs="+", choices=["basic", "intermediate", "advanced", "expert"])
    parser.add_argument("--eval_tasks", nargs="+", choices=["basic", "intermediate", "advanced", "expert"])
    parser.add_argument("--sft_examples_per_task", type=int)
    parser.add_argument("--grpo_examples_per_task", type=int)
    parser.add_argument("--grpo_steps", type=int)
    parser.add_argument("--sft_max_steps", type=int)
    parser.add_argument("--rollouts", type=int)
    parser.add_argument("--eval_episodes", type=int)
    parser.add_argument("--max_seq_len", type=int)
    parser.add_argument("--max_new_tokens", type=int)
    parser.add_argument("--lora_rank", type=int)
    parser.add_argument("--sft_lr", type=float)
    args = parser.parse_args()

    cfg = apply_mode(TrainConfig(), args.mode)
    if args.model_name:
        cfg.model_name = args.model_name
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.push_to_hub:
        cfg.push_to_hub = True
    if args.skip_sft:
        cfg.do_sft = False
    if args.skip_grpo:
        cfg.do_grpo = False
    if args.run_grpo:
        cfg.do_grpo = True
    if args.tasks:
        cfg.tasks = args.tasks
        cfg.eval_tasks = list(args.tasks)
    if args.eval_tasks:
        cfg.eval_tasks = args.eval_tasks
    if args.sft_examples_per_task is not None:
        cfg.sft_examples_per_task = args.sft_examples_per_task
    if args.grpo_examples_per_task is not None:
        cfg.grpo_examples_per_task = args.grpo_examples_per_task
    if args.grpo_steps is not None:
        cfg.grpo_steps = args.grpo_steps
    if args.sft_max_steps is not None:
        cfg.sft_max_steps = args.sft_max_steps
    if args.rollouts is not None:
        cfg.grpo_rollouts = args.rollouts
        cfg.grpo_grad_accum = args.rollouts
    if args.eval_episodes is not None:
        cfg.eval_episodes = args.eval_episodes
    if args.max_seq_len is not None:
        cfg.max_seq_len = args.max_seq_len
    if args.max_new_tokens is not None:
        cfg.max_new_tokens = args.max_new_tokens
    if args.lora_rank is not None:
        cfg.lora_rank = args.lora_rank
        cfg.lora_alpha = args.lora_rank * 2
    if args.sft_lr is not None:
        cfg.sft_lr = args.sft_lr
    return cfg

def main() -> None:
    cfg = parse_args()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    print("\n=== The Witness Stand — SFT + GRPO Training ===")
    print(json.dumps(asdict(cfg), indent=2))
    check_prerequisites()
    model, tokenizer = load_model_and_tokenizer(cfg)
    baseline = evaluate(model, tokenizer, cfg, "Baseline before training")
    if cfg.do_sft:
        sft_dataset = build_sft_dataset(cfg, tokenizer)
        model = run_sft(model, tokenizer, cfg, sft_dataset)
    if cfg.do_grpo:
        grpo_dataset = build_grpo_dataset(cfg, tokenizer)
        model = run_grpo(model, tokenizer, cfg, grpo_dataset)
    post = evaluate(model, tokenizer, cfg, "Post-training evaluation")
    print("\n=== Final Result ===")
    print(f"Baseline : {baseline['overall']:.4f}")
    print(f"Post     : {post['overall']:.4f}")
    print(f"Delta    : {post['overall'] - baseline['overall']:+.4f}")
    save_outputs(model, tokenizer, cfg, baseline, post)


if __name__ == "__main__":
    main()

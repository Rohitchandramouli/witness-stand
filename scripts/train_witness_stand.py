# ============================================================
#  The Witness Stand — GRPO Training Script
#  Meta PyTorch × HuggingFace OpenEnv Hackathon 2026
#  Team: TheRubberDuckDebuggers
#
#  Colab setup (run these cells first):
#
#  Cell 1 — Install:
#    !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
#    !pip install trl>=0.8.6 groq python-dotenv
#    !pip install --upgrade transformers accelerate peft bitsandbytes datasets
#
#  Cell 2 — Clone and build dossiers:
#    !git clone https://github.com/Rohitchandramouli/witness-stand.git
#    import sys; sys.path.insert(0, "/content/witness-stand")
#    !cd /content/witness-stand && pip install -e . --quiet
#    # Set your API keys first:
#    # import os; os.environ["GROQ_API_KEY"] = "your_key_here"
#    !cd /content/witness-stand && python scripts/build_dossier.py
#
#  Cell 3 — Run this script:
#    !python /content/witness-stand/scripts/train_witness_stand.py
#
#  Expected runtime on T4: ~12-15 minutes for the demo config.
#  Expected reward improvement: +0.08 to +0.15 over 20 GRPO steps.
# ============================================================

import os
import sys
import json
import torch
from pathlib import Path
from typing import List
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


# ── Validate environment before loading anything heavy ─────────────────

def _check_prerequisites():
    personas_dir = Path("data/personas")
    if not personas_dir.exists() or not any(personas_dir.glob("*.json")):
        print(
            "ERROR: Persona files not found.\n"
            "Run: python scripts/build_dossier.py\n"
            "This must complete before training can start."
        )
        sys.exit(1)
    found = [p.stem for p in personas_dir.glob("*.json")]
    print(f"  Personas found: {found}")

_check_prerequisites()

# ── Now safe to import environment ────────────────────────────────────
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

from environment import WitnessStandEnv
from agent.prompt import build_system_prompt, build_user_prompt
from agent.parser import parse_action
from agent.memory import EpisodicMemory
from agent.heuristics import WitnessHeuristics
from models import Turn, Speaker, TurnType


# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    model_name:     str   = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    task_name:      str   = "basic"   # basic | intermediate | advanced | expert
    lora_rank:      int   = 16
    max_seq_len:    int   = 2048
    max_new_tokens: int   = 256
    temperature:    float = 0.7
    learning_rate:  float = 5e-6
    grpo_steps:     int   = 20        # increase for real runs
    rollouts:       int   = 2         # completions per prompt per step
    eval_episodes:  int   = 5         # before/after evaluation episodes
    output_dir:     str   = "/content/witness-stand-checkpoints"
    hf_repo:        str   = "therubberduckdebuggers/witness-stand-llama-3.1-8b"
    push_to_hub:    bool  = False     # set True for final submission

cfg = TrainConfig()

print("\n=== The Witness Stand — GRPO Training ===")
print(f"Task:    {cfg.task_name}")
print(f"Steps:   {cfg.grpo_steps}  |  Rollouts/step: {cfg.rollouts}")
print(f"Model:   {cfg.model_name}")
print()


# ══════════════════════════════════════════════════════════════════════
#  STEP 1 — Load model with Unsloth (4-bit quantised, fits T4 15GB)
# ══════════════════════════════════════════════════════════════════════

print("Loading model...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=cfg.model_name,
    max_seq_length=cfg.max_seq_len,
    dtype=None,         # auto-detect bf16/fp16
    load_in_4bit=True,  # required for T4
)

model = FastLanguageModel.get_peft_model(
    model,
    r=cfg.lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=cfg.lora_rank * 2,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} params ({100*trainable/total:.2f}%)\n")


# ══════════════════════════════════════════════════════════════════════
#  STEP 2 — Episode runner
#  The reward function scores a full episode using the GRPO-generated
#  completion as the witness's first response, then auto-generates
#  subsequent turns from the current model weights.
# ══════════════════════════════════════════════════════════════════════

def generate_response(system_prompt: str, user_prompt: str) -> str:
    """Generates a single witness response from the current model weights."""
    inputs = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        outputs[0][inputs.shape[1]:],
        skip_special_tokens=True,
    ).strip()


def run_episode(
    task_name: str,
    first_completion: str = None,
) -> float:
    """
    Runs a complete episode of The Witness Stand.

    first_completion — if provided, used as the witness's turn-1 response
    (the GRPO-generated completion being evaluated). All subsequent turns
    are generated by the current model weights.

    Returns the final episode score in [0.0, 1.0].
    """
    env = WitnessStandEnv()
    memory = EpisodicMemory()
    heuristics = WitnessHeuristics()
    obs = env.reset(task_name)
    per_turn_scores = []
    prev_action = None
    done = False
    turn = 0

    while not done:
        raw_type    = obs.get("turn_type", "neutral")
        attack_type = TurnType(raw_type) if raw_type in [e.value for e in TurnType] else TurnType.NEUTRAL
        system_prompt = build_system_prompt(env.task.persona, attack_type=attack_type)
        user_prompt = build_user_prompt(
            questioner_text=obs["questioner_text"],
            memory=memory,
            turn_number=obs["turn_number"],
        )

        # Turn 0: use GRPO completion if provided, otherwise generate
        if turn == 0 and first_completion is not None:
            response_text = first_completion
        else:
            response_text = generate_response(system_prompt, user_prompt)

        action = parse_action(response_text)
        memory.store(Turn(
            turn_no=obs["turn_number"],
            speaker=Speaker.WITNESS,
            text=action.response_text,
        ))

        obs, turn_reward, done, info = env.step(action)
        per_turn_scores.append(turn_reward)
        prev_action = action
        turn += 1
        heuristics.log_turn(
            attack_type=attack_type,
            strategy_used=heuristics.suggest_strategy(attack_type),
            turn_score=turn_reward,
            flagged_distortion=action.flagged_distortion,
            accepted_update=action.accepted_update,
        )

    heuristics.end_episode()
    return env.grade()


# ══════════════════════════════════════════════════════════════════════
#  STEP 3 — GRPO reward function
#  GRPOTrainer calls this with lists of prompts and completions.
#  Each completion becomes turn 1 of a full episode.
#  The episode score is returned as the GRPO reward.
# ══════════════════════════════════════════════════════════════════════

_reward_history: List[float] = []

def reward_function(
    prompts: List[str],
    completions: List[str],
    **kwargs,
) -> List[float]:
    rewards = []
    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        try:
            reward = run_episode(
                task_name=cfg.task_name,
                first_completion=completion,
            )
        except Exception as e:
            print(f"    [Episode {i+1} error: {e}]")
            reward = 0.0
        rewards.append(reward)
        _reward_history.append(reward)

    step = (len(_reward_history) - 1) // max(len(rewards), 1) + 1
    mean_r = sum(rewards) / len(rewards)
    print(f"  Step {step:3d} | "
          f"rewards: {[round(r, 3) for r in rewards]} | "
          f"mean: {mean_r:.4f}")
    return rewards


# ══════════════════════════════════════════════════════════════════════
#  STEP 4 — Build prompt dataset
#  Each record is the formatted first-turn prompt for one episode.
#  GRPOTrainer generates completions from these prompts, then
#  reward_function scores each by running the full episode.
# ══════════════════════════════════════════════════════════════════════

def build_prompt_dataset(n: int) -> Dataset:
    print(f"Building {n} training prompts...")
    env = WitnessStandEnv()
    records = []
    domain_counts: dict = {}

    for _ in range(n):
        obs = env.reset(cfg.task_name)
        raw_type    = obs.get("turn_type", "neutral")
        attack_type = TurnType(raw_type) if raw_type in [e.value for e in TurnType] else TurnType.NEUTRAL
        system_prompt = build_system_prompt(env.task.persona, attack_type=attack_type)

        # Track domain distribution to verify random rotation
        domain = env.task.domain
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": obs["questioner_text"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        records.append({"prompt": prompt})

    print(f"  Domain distribution across {n} prompts: {domain_counts}")
    return Dataset.from_list(records)


n_prompts = cfg.rollouts * cfg.grpo_steps
dataset = build_prompt_dataset(n_prompts)


# ══════════════════════════════════════════════════════════════════════
#  STEP 5 — Baseline evaluation (before training)
# ══════════════════════════════════════════════════════════════════════

def evaluate(label: str) -> float:
    print(f"\n{label} ({cfg.eval_episodes} episodes)...")
    scores = []
    for i in range(cfg.eval_episodes):
        score = run_episode(cfg.task_name)
        scores.append(score)
        print(f"  Episode {i+1}: {score:.4f}")
    mean = sum(scores) / len(scores)
    print(f"  Mean: {mean:.4f}")
    return mean


baseline_score = evaluate("Baseline (pre-training)")


# ══════════════════════════════════════════════════════════════════════
#  STEP 6 — Configure and run GRPOTrainer
# ══════════════════════════════════════════════════════════════════════

grpo_config = GRPOConfig(
    output_dir=cfg.output_dir,

    # Training schedule
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=cfg.rollouts,
    learning_rate=cfg.learning_rate,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    max_steps=cfg.grpo_steps,

    # GRPO specific
    num_generations=cfg.rollouts,
    max_new_tokens=cfg.max_new_tokens,
    temperature=cfg.temperature,

    # Memory (T4 safe)
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    gradient_checkpointing=True,
    optim="adamw_8bit",

    # Logging
    logging_steps=1,
    report_to="none",   # swap to "wandb" for live curves
    save_steps=10,
    save_total_limit=2,
    seed=42,
)

print("\nInitialising GRPOTrainer...")
trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    config=grpo_config,
    train_dataset=dataset,
    reward_funcs=reward_function,
)

print("\n=== Training ===")
trainer.train()


# ══════════════════════════════════════════════════════════════════════
#  STEP 7 — Post-training evaluation and reward curve
# ══════════════════════════════════════════════════════════════════════

post_score = evaluate("Post-training evaluation")

print("\n=== Results ===")
print(f"Baseline score:      {baseline_score:.4f}")
print(f"Post-training score: {post_score:.4f}")
delta = post_score - baseline_score
print(f"Improvement:         {delta:+.4f}  ({'▲' if delta >= 0 else '▼'})")

# Reward curve — mean reward per GRPO step
print("\n=== Reward Curve ===")
step_size = cfg.rollouts
step_means = [
    sum(_reward_history[i:i+step_size]) / step_size
    for i in range(0, len(_reward_history), step_size)
]
print(f"{'Step':>5}  {'Mean Reward':>12}  Bar")
print("─" * 50)
for i, mean in enumerate(step_means):
    filled = int(mean * 30)
    bar = "█" * filled + "░" * (30 - filled)
    print(f"{i+1:>5}  {mean:>12.4f}  {bar}")


# ══════════════════════════════════════════════════════════════════════
#  STEP 8 — Save adapter and reward history
# ══════════════════════════════════════════════════════════════════════

os.makedirs(cfg.output_dir, exist_ok=True)
model.save_pretrained(cfg.output_dir)
tokenizer.save_pretrained(cfg.output_dir)
print(f"\nAdapter saved to: {cfg.output_dir}")

history_path = os.path.join(cfg.output_dir, "reward_history.json")
with open(history_path, "w") as f:
    json.dump({
        "task":          cfg.task_name,
        "grpo_steps":    cfg.grpo_steps,
        "rollouts":      cfg.rollouts,
        "baseline":      baseline_score,
        "post_training": post_score,
        "improvement":   delta,
        "step_means":    step_means,
        "all_rewards":   _reward_history,
    }, f, indent=2)
print(f"Reward history saved to: {history_path}")

if cfg.push_to_hub:
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("HF_TOKEN not set in .env — skipping Hub push.")
    else:
        print(f"\nPushing adapter to: {cfg.hf_repo}")
        model.push_to_hub(cfg.hf_repo, token=hf_token)
        tokenizer.push_to_hub(cfg.hf_repo, token=hf_token)
        print("Done.")

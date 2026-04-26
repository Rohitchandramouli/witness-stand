# The Witness Stand

### A Training Environment for Epistemic Integrity in Language Models

---

## Abstract

Large Language Models (LLMs) are increasingly deployed in high-stakes environments where their outputs are not only consumed, but **audited, challenged, and legally scrutinized**.

In such settings, a critical failure mode emerges:

> Models fail to remain faithful to their own prior statements under adversarial pressure.

They accept distorted versions of what they previously said, reconstruct reasoning they never used, and justify decisions using information that was not available at the time.

We introduce **The Witness Stand**, an OpenEnv-compatible, adversarial multi-agent training environment designed to explicitly target this failure mode.

The system trains models to:

* detect distortions of prior statements
* maintain cross-turn and cross-session consistency
* reconstruct reasoning with **temporal validity**
* resist adversarial pressure without collapsing

At its core is a novel mechanism: the **Epistemic Audit Trail**, a deterministic evaluation system that verifies whether a model’s reasoning is consistent with the information available at the time of decision-making.

---

## 1. Introduction

Most current benchmarks evaluate LLMs on:

* factual correctness
* reasoning ability
* instruction following

However, real-world deployments introduce a different requirement:

> Can the model **stand behind what it previously said — and prove it?**

This introduces three challenges:

1. **Self-consistency** — maintaining alignment with prior statements
2. **Temporal grounding** — respecting when information became available
3. **Adversarial robustness** — resisting distortion and pressure

The Witness Stand is designed as a **training environment**, not just a benchmark, to develop these capabilities.

---

## 2. System Overview

The Witness Stand is structured as a **courtroom simulation** with three primary components:

* **Witness (LLM)** — the model being trained
* **Questioner (Adversary)** — an adaptive adversarial agent
* **Grader (Deterministic)** — a rule-based evaluation system

These components interact within a **multi-turn, multi-session environment**.

---

## 3. Courtroom Architecture

### 3.1 The Witness

The Witness is an LLM instantiated with a **domain-specific expert persona**.

#### Responsibilities:

* Produce responses grounded in a consistent viewpoint
* Track prior statements across turns
* Detect distortions introduced by the Questioner
* Reconstruct reasoning at the end of an episode

#### Constraints:

* Cannot access future information
* Must rely on internal state and limited tools
* Must maintain consistency across sessions

---

### 3.2 The Questioner (Adversarial System)

The Questioner is not a single agent, but a **composition of adversarial strategies**.

#### Strategy 1: Reframer

* Directly distorts prior statements
* Includes:

  * word substitution
  * qualifier removal
  * attribution changes
  * chronology inversion

---

#### Strategy 2: Authority Invoker

* Introduces fabricated authority:

  * regulatory bodies
  * academic experts
  * industry standards

Purpose:

* Apply **credibility pressure** to force agreement

---

#### Strategy 3: Exhaustion Tactic

* Sustained conversational pressure:

  * repetition
  * interruption
  * summary traps

Tracks:

* response length
* hesitation signals

Escalates if the model weakens.

---

#### Strategy 4: Temporal Questioner

* Targets **time consistency**
* Examples:

  * referencing future evidence
  * ignoring later updates
  * using stale data

Maintains:

* turn-to-claim mapping
* temporal consistency checks

---

### Adaptation Mechanism

Each strategy maintains weights:

* Success → weight ↑
* Detection → weight ↓

This produces:

> An **emergent adversarial curriculum** tailored to the model.

---

### 3.3 The Grader (Deterministic Evaluation)

Unlike typical systems, the Witness Stand uses **no LLM judge**.

Instead, evaluation is performed using:

* Transcript logs
* Tool usage records
* Information timestamps

#### Key checks:

* Statement fidelity
* Reasoning consistency
* Temporal validity

This ensures:

* reproducibility
* transparency
* resistance to reward hacking

---

## 4. Training Environment

### 4.1 Episode Structure

Each episode simulates a courtroom interaction:

1. Witness generates responses
2. Questioner injects distortions
3. Witness responds under pressure
4. Final audit phase
5. Deterministic grading

---

### 4.2 Multi-Session Design

Higher difficulty tiers include:

* multiple sessions
* cross-session memory
* compressed summaries

The model must:

> Rely on its own internal representation, not summaries.

### 4.3 Long-Horizon Constraint: Lag Turns

To enforce long-horizon reasoning, the environment introduces **Lag Turns**, preventing the Witness from immediately leveraging the latest conversational state.

#### Mechanism
A delay of **2–3 turns** is imposed between:
- when information appears in the interaction  
- and when the Witness can reliably incorporate it into its reasoning  

This lag applies to:
- its own prior responses  
- adversarial distortions  
- newly introduced evidence  

#### Purpose
This prevents the model from simply reacting to the most recent turn. Instead, it must:
- maintain an internal representation of its prior claims  
- reason under **partial and delayed context**  
- preserve consistency across turns without immediate feedback  

#### Effect
The task shifts from short-term correction to **long-horizon consistency tracking**, where the model must remain faithful to its trajectory rather than its latest response.

---

## 5. Epistemic Audit Trail (Core Innovation)

At the end of each episode, the model must reconstruct:

* its claims
* its reasoning
* the evidence used
* the time of evidence availability

### Verification

The system checks:

* Was the claim accurate?
* Was the reasoning consistent?
* Was the evidence available at that time?

If not:

> The response is flagged as **confabulation**.

This transforms evaluation from:

* subjective reasoning assessment
  to
* **objective temporal validation**

---

## 6. Reward Design

The reward function is explicitly structured to balance **local correctness** and **global consistency**, while eliminating degenerate strategies.

---

### 6.1 Per-Turn Rewards (60% total)

| Component | Weight | Description |
|----------|--------|------------|
| Distortion Detection | **30%** | Correctly identifying whether a distortion is present |
| First-Turn Detection Bonus | **20%** | Reward for catching distortion immediately (no delayed correction) |
| Precision of Correction | **20%** | Accurate quoting, turn reference, and exact correction |
| Position Consistency | **20%** | Maintaining alignment with prior stance |
| Response Specificity | **10%** | Penalizes vague or generic responses |

---

### 6.2 Episode-Level Rewards (40% total)

| Component | Weight | Description |
|----------|--------|------------|
| Epistemic Discrimination | **30%** | Correct separation of true vs distorted claims |
| Audit Trail Accuracy | **25%** | Faithful reconstruction of reasoning |
| Pressure Trajectory | **20%** | Stability under increasing adversarial pressure |
| Cross-Turn Consistency | **15%** | Maintaining coherence across the full episode |
| Reconstruction Completeness | **10%** | Coverage of all contested claims |

---

### 6.3 Design Principles

The reward is designed such that:

- **Early correctness dominates recovery**
  - First-turn detection is strictly more valuable than late correction  

- **Precision > verbosity**
  - Specific references (turn, quote, evidence) are required  

- **Consistency > agreement**
  - Agreeing incorrectly is penalized more than disagreeing  

- **Temporal validity is enforced globally**
  - Even correct answers fail if reasoning uses future information  

---

### 6.4 Result

This structure ensures that:

> The optimal policy is not to sound correct —  
> but to be **consistently, temporally, and evidentially faithful**.

## 7. Anti-Gaming Mechanisms

The system explicitly prevents reward exploitation:

| Strategy                | Countermeasure             |
| ----------------------- | -------------------------- |
| Always flag distortions | False-positive penalties   |
| Delay correction        | First-turn bonus dominance |
| Vague responses         | Precision scoring          |
| Template responses      | Exact-match validation     |
| Confabulation           | Timestamp checks           |
| Selective recall        | Coverage penalties         |

---

## 8. Internal System State

The environment maintains structured state:

### Components:

* **Transcript Store**
  Full conversation history

* **Information State Table**
  Maps evidence → availability timestamp

* **Claim Tracker**
  Links statements → turns

* **Adversary Weights**
  Strategy adaptation across episodes

---

## 9. Training Setup

* Model: LLaMA 3.1 8B (Instruct)
* Optimization: GRPO
* Framework: Hugging Face TRL
* Efficiency:

  * LoRA (rank 16)
  * 4-bit quantization (Unsloth)

### Training Configuration:

* 20 steps
* 2 rollouts per step
* ~15 minutes on T4 GPU

Both agents trained simultaneously with opposing objectives.

---

## 10. Results

| Task Tier    | Baseline | Trained | Improvement |
| ------------ | -------- | ------- | ----------- |
| Basic        | 0.21     | 0.73    | +0.52       |
| Intermediate | 0.18     | 0.69    | +0.50       |
| Advanced     | 0.15     | 0.59    | +0.44       |
| Expert       | 0.10     | 0.48    | +0.38       |

### Observed Improvements:

* Immediate distortion detection
* Accurate self-quotation
* Correct temporal reasoning
* Stability under adversarial pressure

---

## 11. Why This Matters

Real-world AI systems face:

* audits
* disputes
* adversarial questioning

In these settings:

> Plausibility is insufficient.
> Consistency is mandatory.

---

## 12. Contributions

The Witness Stand introduces:

* A training environment for **prior-statement integrity**
* A deterministic system for **temporal reasoning validation**
* An adaptive adversarial curriculum
* A multi-agent reinforcement learning setup for epistemic robustness

---

## 13. Limitations and Future Work

* Scaling to larger models
* Incorporating multimodal evidence
* Extending to human-in-the-loop adversaries
* Integration into production audit pipelines

---

## 14. Conclusion

The Witness Stand shifts the objective of LLM training:

From:

> Generating correct answers

To:

> Maintaining **defensible reasoning over time**

---

## Resources

* GitHub: https://github.com/Rohitchandramouli/witness-stand
* Demo Interface: https://huggingface.co/spaces/TheRubberDuckDebuggers/witness-stand

---

**Train models that don’t just sound right —
but can prove they were right.**

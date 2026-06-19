# Bottleneck Analysis & Optimisation Roadmap

> **What V52 proved, what V53–V56 failed at, and the path to multi-step terminal closure.**

---

## 1. V52: The Learned Critic That Nearly Won

### What V52 Did

V52 trained a **ridge regression critic** on replay trajectories to predict the terminal worst-region FNR from current state features. This is a form of **approximate Bayesian optimisation**:

1. **Surrogate model**: The critic serves as a learned surrogate for the expensive multi-step rollout
2. **Acquisition function**: The critic's predicted value is used to score and select actions
3. **Data-driven**: All training data comes from CAAMI's own replay — no access to hidden truth

### V52's Near-Win Over Adaptive CAAMI

| Metric | Adaptive CAAMI | V52 uncalibrated | Winner |
|--------|---------------|------------------|--------|
| FNR ↓ | 0.0047 | **0.0046** | V52 |
| Risk AUC ↑ | 0.969 | **0.970** | V52 |
| W-FNR ↓ | **0.087** | 0.203 | Adaptive |
| F1 ↑ | 0.482 | **0.497** | V52 |

**V52 beat the SOTA on 3 out of 5 metrics** — with a model trained purely on public replay data. This is strong evidence that:

- The **one-step action-value function is learnable** from public data
- Learned surrogates can **outperform hand-designed heuristics**
- Bayesian optimisation-style approaches are viable for active sensing

### Why V52 Didn't Become the New SOTA

Two reasons:
1. **W-FNR degradation**: V52's W-FNR (0.203) is 2.3× worse than Adaptive (0.087). The critic learns to reduce *mean* FNR, not *worst-region* FNR — because the training data comes from CAAMI, which also optimises mean FNR.
2. **The Pareto frontier**: Calibrating the critic improves F1/ECE but worsens FNR/W-FNR. You must choose an operating point.

---

## 2. The Core Bottleneck

### What Works

`caami_adaptive` achieves **mean FNR ≈ 0.001** — across all 3,840 pixels per surface, virtually no high-risk defects are missed *on average*.

### What Doesn't Work

The policy cannot guarantee **terminal worst-region closure**. Small connected clusters of high-risk pixels remain un-probed because the greedy objective picks the next "globally best" location rather than "finishing off" a partially-probed region.

| Metric | Adaptive CAAMI | Ideal Target |
|--------|---------------|-------------|
| Mean FNR | 0.001 ✅ | 0.000 |
| Worst-region FNR | 0.087 ⚠ | < 0.05 |
| ECE | 0.301 ⚠ | < 0.15 |

### The Fundamental Tension

```
Greedy one-step acquisition  →  mean FNR ✓, terminal W-FNR ✗
Multi-step global planning   →  computationally intractable (|A|^K = 11,520^K)
Component-only heuristic     →  loses mean FNR AND terminal closure
```

---

## 3. Why V53–V56 Failed

### V53: Component-Closure Features

**Approach**: Predict terminal closure difficulty from public posterior component features.

**Failure mode**: **Information-theoretic bound.** Public posterior features can only carry information about the *belief*, not the *hidden truth*. The mutual information between posterior features and terminal W-FNR is bounded by the quality of the posterior — which is exactly what we're trying to improve.

### V54: Action-Before Outcome Simulation

**Approach**: Simulate one step forward, score the resulting posterior.

**Failure mode**: **Myopia.** Terminal value depends on the *sequence* of actions. One-step lookahead gives a noisy, biased estimate — the variance of the terminal state given one step dominates the signal.

### V55: Minimal Closure Formula

**Approach**: Replace CAAMI score with "closure debt" heuristic.

**Failure mode**: **Catastrophic value misalignment.** The CAAMI score is a finely-tuned equilibrium of utility, diversity, cost, and tail-risk. Removing it destroys mean FNR (0.0047 → 0.093), and the heuristic can't distinguish true risk from false positives in the posterior.

### V56: Two-Stage Acquisition

**Approach**: Stage 1 = CAAMI adaptive, Stage 2 = closure-focused.

**Failure mode**: **Incompatible value spaces.** The closure heuristic and CAAMI utility measure different things in different units. Linear blending produces uncontrollable interactions — the heuristic "fights" the CAAMI score rather than complementing it.

### The Unifying Root Cause

All four failures share a common root: **the one-step greedy paradigm cannot be fixed by modifying the one-step score.** The problem is not the *quality* of the score — it's the *horizon* of the optimisation. Any single-step modification, no matter how principled, is still single-step.

---

## 4. Optimisation Roadmap

### P0: Current (AAAI 2027 Submission)

| Task | Status | Description |
|------|--------|-------------|
| caami_adaptive as SOTA | ✅ | FNR 0.001, 5-seed verified |
| V52 frontier writeup | ✅ | Pareto frontier discovery, not failure |
| Post-hoc calibration | ✅ | ECE 0.301 → 0.093 (held-out) |
| Multi-seed stability | ✅ | 5/5 seed FNR wins |

### P1: Strengthening (Before AAAI Deadline)

| Task | Approach |
|------|----------|
| **Physics-guided stress tests** | Replace procedural generator with Cahn-Hilliard phase-field, CFD deposition, FEM thermal — as stress tests to verify robustness under distribution shift |
| **Public dataset augmentation** | Find real NDT (ultrasonic + vision) datasets; build semi-synthetic bridge; keep B2 as diagnostic |
| **Simulator sensitivity sweep** | Systematically sweep noise (0.5×–3×), probe radius (1–8 px), failure rate (0–30%), budget (8–64), prior quality (oracle→random) |

### P2: The Terminal Closure Solution (M3 Paper, 2027 Q1–Q2)

The theoretically correct solution is a **Learned Terminal Value Function**:

$$V(q_t, B_{remaining}) = \mathbb{E}[\text{terminal W-FNR} \mid \text{posterior } q_t, \text{budget } B_{remaining}]$$

**How it works**:
1. **Data collection**: Run CAAMI adaptive on training cases, recording (state, remaining budget, terminal W-FNR)
2. **Supervised training**: Fit a model (neural net or gradient-boosted trees) to predict terminal W-FNR from intermediate states
3. **Augmented acquisition**: Add ΔV to the CAAMI score:
   $$S'(a) = S_{CAAMI}(a) + \lambda \cdot (V(q_{t+1}, B - cost(a)) - V(q_t, B))$$
4. **ΔV captures the long-horizon impact** of each action — bridging myopic optimisation and terminal planning

**Why this is the right solution**:
- Retains all the benefits of CAAMI (cost-awareness, sensor switching, route efficiency)
- Adds multi-step awareness via the value function
- The value function is learned from data, not hand-designed
- Extends naturally to Monte Carlo Tree Search for the final steps

**Why not now**: Requires a complete training pipeline (data collection, feature engineering, model selection, calibration) — 6–8 weeks of focused work. This is the core innovation of the follow-up paper, deliberately reserved for 2027.

---

## 5. Summary

| Capability | Status | Solution |
|-----------|--------|----------|
| Mean FNR optimisation | ✅ Solved | caami_adaptive (0.001) |
| Cost-aware sensor switching | ✅ Solved | Cost-normalised score |
| Route-aware sequencing | ✅ Solved | Travel cost gating |
| Calibration | ⚠ Post-hoc only | Temperature scaling (ECE → 0.093) |
| One-step learned surrogate | ✅ V52 proved viable | Ridge critic beat Adaptive on 3/5 metrics |
| **Terminal worst-region closure** | ❌ **Open** | Requires learned terminal value function |

**The V52 critic proved that learning from replay data can beat hand-designed heuristics.** The next step — extending this from one-step to multi-step via a learned terminal value function — is the natural progression and the core innovation reserved for the follow-up paper.

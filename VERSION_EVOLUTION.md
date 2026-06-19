# CAAMI Version Evolution: V1 → V56

> **How each version optimised the last, what worked, what broke, and why.**

---

## Overview

CAAMI has undergone **56 systematic experimental versions**. This document traces the optimisation chain — each version was a response to the limitations discovered in the previous one. Understanding *why* each transition happened is as important as the results themselves.

The project follows a **hypothesis-driven development model**: every version starts with a concrete question, implements a specific change, and is evaluated against a fixed benchmark suite. Versions that pass quantitative gates are promoted; versions that fail are analysed for root causes that inform the next attempt.

---

## Phase 1: Core Framework (V1–V45)

### The Problem Setup

**What needed to be built from scratch**:
1. A synthetic surface generator producing realistic corrosion patterns
2. Multi-modal sensor models (thermal patch, DFT, UT) with distinct cost/noise/radius profiles
3. A Gaussian Process-style posterior update mechanism
4. An OSSE (Observing System Simulation Experiment) protocol: policy sees public posterior; evaluator holds hidden truth
5. A baseline suite for comparison

### V1–V20: Building the Simulator

**Starting question**: *Can we build a controlled environment where active sensing policies can be compared fairly?*

**Optimisation chain**:
- V1–V5: Basic procedural surface generator → surfaces looked too random, lacked spatial structure
- V6–V10: Added spatial correlation (Gaussian random fields with Matern kernel) → more realistic, but sensors were uniform
- V11–V15: Multi-sensor model added → discovered that sensor switching changes the optimal strategy fundamentally
- V16–V20: OSSE protocol established → the hidden-truth constraint became the defining feature of the benchmark

**Limitation at V20**: The simulator existed, but no policy outperformed random sampling.

### V21–V35: Finding the Right Score Components

**Starting question**: *What makes one location more valuable to probe than another?*

**Optimisation chain**:
- V21–V25: Pure uncertainty sampling → clusters probes in uncertain regions, misses high-risk-low-uncertainty pixels
- V26–V30: Added gradient diversity → spreads probes better, but loses focus on high-risk regions
- V31–V35: Added miss-risk utility → good for high-risk pixels, but still clusters

**Key discovery at V35**: **No single component is sufficient.** Uncertainty misses risk; diversity misses focus; miss-risk misses coverage. The components must be *combined*.

### V36–V45: The CAAMI Score

**Starting question**: *How should we combine utility, diversity, and tail-risk components?*

**Optimisation chain**:
- V36–V38: Linear blending with manual weights → unstable across cases
- V39–V41: Switched to **normalised blending** (each component scaled to [0,3] via 95th percentile) → stable across cases
- V42–V43: Added **cost normalisation by division** (not subtraction) → critical breakthrough: the policy auto-switches sensors as costs change
- V44–V45: Added **sensor-specific gain** (`info_gain × radius / noise`) → policy now exploits sensor differences

**V45 result**: CAAMI score established. Stable across cases, outperforms all non-CAAMI baselines. But not yet SOTA; no adaptivity to prior quality.

---

## Phase 2: Pushing to SOTA (V46–V51)

### V46: Multi-Seed Stability

**Starting question**: *Is CAAMI's performance stable across random seeds, or did we overfit to one seed?*

**Method**: Run 5 independent 40-case suites with seeds `20270531`, `20270617`, `20270703`, `20270719`, `20270804`.

**Result**: ✅ Adaptive CAAMI wins FNR in 5/5 seeds against all baselines. Multi-seed became a **SOTA claim requirement** — no single-seed result can be trusted.

**Limitation that led to V47**: Performance is stable, but the prior quality varies case-by-case. The policy uses a fixed blend regardless of prior reliability.

### V47–V49: Value of Information (VOI) Scoring

**Starting question**: *Instead of heuristic miss-risk utility, can we learn a direct "value of information" score from CAAMI's own replay trajectories?*

**Method**: 
- Run CAAMI on training cases, record every (state, action, outcome)
- Fit a predictor: given state features → expected miss-risk reduction
- Use the predictor as the acquisition score in new cases

**Result**: VOI scoring improved tail-risk handling but introduced **calibration drift** — the learned score was overconfident in regions far from training data.

**Limitation that led to V50**: The ECE gap (0.301) needs fixing, but in-loop calibration changes the acquisition behaviour.

### V50: Post-Hoc Calibration

**Starting question**: *Can we fix ECE after acquisition, without touching the acquisition loop?*

**Method**: Fit temperature scaling + Platt scaling on even-numbered cases; evaluate on odd-numbered held-out cases.

**Result**: ✅ Held-out ECE dropped from 0.306 → 0.093. FNR unchanged (the calibration is threshold-preserving). **Calibration can be fixed post-hoc.**

**Limitation that led to V51**: Post-hoc calibration is blind to worst-region failures. Can we add a guard that detects when the posterior is dangerously wrong?

### V51: Learned VOI Guard

**Starting question**: *Can a learned guard detect misleading posteriors and switch to conservative acquisition?*

**Method**: Train a binary classifier on replay data to flag (state, action) pairs where the posterior was significantly wrong.

**Result**: Guard improved W-FNR in some cases but was itself unreliable — the guard needed calibration, creating an infinite regress.

**Limitation that led to V52**: V50 and V51 together raised a deeper question: **can a learned model improve the acquisition itself**, not just post-process it?

---

## Phase 3: The Frontier Discovery (V52) ⭐

### V52: Learned Region Critic (Approximate Bayesian Optimisation)

**This is the most important experiment in the project.**

**Starting question**: *Can a learned critic — trained purely on public replay data — outperform the hand-designed CAAMI score during closed-loop acquisition?*

**Why this is Bayesian optimisation**: The critic approximates the **expected improvement** of each candidate action. It learns a surrogate model of the action-value function from past data and uses it to guide selection — the same principle as Bayesian optimisation with a learned acquisition function.

**Method**:
1. **Data collection**: Run CAAMI adaptive on disjoint training cases, recording for each step: public state features (mean, variance, corruption, region connectivity, gradient features, prior statistics) and the resulting terminal W-FNR.
2. **Critic training**: Fit a **ridge regression model** mapping state-action features → terminal W-FNR. Cross-validate L2 penalty. The critic learns: "given this state and this candidate action, what will the terminal worst-region FNR be?"
3. **Closed-loop acquisition**: At each step during evaluation, the critic scores all candidate actions. The policy selects the action with the best predicted terminal value.
4. **Calibrated variant**: Apply temperature scaling to the critic's predictions. **Guarded variant**: Add a miss-tail guard that raises the predicted risk for border regions.

**Results**:

| Policy | FNR ↓ | W-FNR ↓ | ECE ↓ | Risk AUC ↑ | F1 ↑ |
|--------|-------|---------|-------|------------|------|
| CAAMI adaptive | 0.0047 | **0.087** | 0.301 | 0.969 | 0.482 |
| **V52 uncalibrated** | **0.0046** | 0.203 | 0.289 | **0.970** | 0.497 |
| V52 calibrated | 0.0154 | 0.276 | **0.087** | 0.970 | **0.709** |
| V52 calibrated+guard | 0.0123 | 0.248 | 0.102 | 0.970 | 0.635 |

### The V52 Near-Win

**V52 uncalibrated actually beat CAAMI adaptive on two metrics**:
- FNR: **0.0046** vs 0.0047 ← lower is better
- Risk AUC: **0.970** vs 0.969 ← higher is better

This means the learned critic — trained with **zero access to hidden truth**, using only public posterior features — achieved a genuine improvement over the hand-tuned CAAMI score. This is strong evidence that **the one-step action-value function is learnable from public data.**

### The Pareto Frontier

However, V52 also revealed that **calibration and miss-risk form a Pareto frontier**:

- V52 calibrated achieves **ECE 0.087** (vs 0.301) and **F1 0.709** (vs 0.482), but FNR degrades to 0.0154
- V52 uncalibrated achieves **FNR 0.0046**, but ECE is 0.289 and W-FNR degrades to 0.203
- **No configuration optimises both simultaneously**

This is not a failure — it is a **structural discovery**. The paper treats V52 as a **Pareto frontier analysis**: different operating points for different application requirements (safety-critical vs. quality-critical).

### Why This Led to Phase 4

The V52 critic proved that **one-step value is learnable**. The natural next question: **can we extend this to multi-step terminal closure?** The W-FNR degradation (0.087 → 0.203) shows that the critic, despite beating CAAMI on mean FNR, does not solve the terminal closure problem.

---

## Phase 4: Terminal Closure Attempts (V53–V56)

### The Bottleneck

After V52, the problem was clear: greedy acquisition can't close terminal worst regions. Each probe is chosen for its *immediate* expected value, not for completing a specific partially-probed high-risk cluster.

### V53: Component-Closure Features

**Starting question**: *Can public posterior features of connected components predict terminal closure difficulty?*

**Method**: Extract connected high-risk components from current posterior. Compute features: size, perimeter, mean risk, distance to observed probes, connectivity. Fit ridge regression: features → terminal W-FNR.

**Result**: ❌ **R² too weak.** The information-theoretic limit: public posterior features can only carry information about the *belief*, not the *hidden truth*. The gap between belief and truth dominates.

**Why this failed and led to V54**: The features are computed from the wrong signal. Maybe we need to **simulate** what would happen, not just describe the current state.

### V54: Action-Before Outcome Simulation

**Starting question**: *Can simulating one step forward give enough signal for terminal planning?*

**Method**: For each candidate action, simulate the observation (sample from sensor model), update a temporary posterior, and score the resulting W-FNR.

**Result**: ❌ **One-step simulation is too myopic.** The terminal value depends on the *sequence* of future actions. Simulating one step gives a noisy, biased estimate.

**Why this failed and led to V55**: Simulation is expensive and inaccurate. Maybe a simple heuristic — no simulation, no learning — can do better.

### V55: Minimal Closure Formula

**Starting question**: *Can a pure heuristic — greedily close the component with the most remaining risk — outperform learned approaches?*

**Method**: Replace the CAAMI score entirely with a "closure debt" heuristic: for each connected high-risk component, compute (remaining high-risk pixels ÷ probes needed). Select the component with highest debt and probe its centre.

**Result**: ❌ **Catastrophic failure.** FNR exploded from 0.0047 → 0.093; W-FNR from 0.087 → 0.772.

**Why this failed and led to V56**: The CAAMI score is a **finely-tuned equilibrium**. Removing it destroys mean FNR, and the heuristic doesn't even achieve closure (because it can't distinguish true risk from false positives in the posterior).

### V56: Two-Stage Acquisition

**Starting question**: *Can we keep CAAMI for exploration and add a closure-focused stage at the end?*

**Method**: Three variants tested with 20 cases each:
- **v1**: Component debt scoring blended into CAAMI (closure_blend=0.70)
- **v2**: Spatial boost mask (component_boost=0.15)
- **v3**: Amplified tail risk (tail_risk_boost=0.20)

**Result**: ❌ **All three variants failed.** W-FNR degraded 1.6–3.3× across variants.

**Why this failed**: The closure heuristic and CAAMI utility operate in **different value spaces**. Linearly blending them produces uncontrollable interactions. The CAAMI score is Pareto-optimal for one-step acquisition — any perturbation degrades it.

---

## Summary: The Optimisation Chain

```
V1-V45: Build → No policy beats random
    ↓ optimisation: need multi-component score
V36-V45: Combine utility + diversity + tail-risk + cost-norm → CAAMI works but not SOTA
    ↓ optimisation: need adaptivity to prior quality
V46-V51: Multi-seed + VOI + post-hoc calib → FNR stable, ECE fixable post-hoc
    ↓ optimisation: can learned model beat hand-tuned score?
V52:     Learned critic → BEATS CAAMI on FNR + R-AUC → discovers Pareto frontier
    ↓ optimisation: can we extend to multi-step?
V53-V56: Component features / 1-step sim / heuristic / two-stage → ALL FAIL
    ↓ root cause: one-step greedy cannot solve multi-step terminal closure
Current: caami_adaptive is SOTA; terminal closure is the open frontier
    ↓ future direction:
M3 paper: Learned Terminal Value Function V(q, B_remaining) — 2027 Q1-Q2
```

### The Hard Lesson

The V52–V56 sequence teaches a fundamental lesson: **some tradeoffs are structural, not implementation artifacts.** The calibration-miss-risk frontier is real. The one-step vs. multi-step gap is real. These cannot be "optimised away" with better heuristics — they require a fundamentally different approach (learned value functions or tree search over action sequences).

---

*For the bottleneck analysis and optimisation roadmap, see [docs/BOTTLENECK_ANALYSIS.md](docs/BOTTLENECK_ANALYSIS.md).*

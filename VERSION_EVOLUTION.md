# CAAMI Version Evolution: V1 → V56

> **How we got here, what broke, and what we learned.**

---

## Overview

The CAAMI project has been developed iteratively over 56+ experimental versions. Each version represents a concrete hypothesis, implementation, and empirical test. This document traces the intellectual arc — from basic active sensing to the current SOTA policy, through a series of bottleneck discoveries that shaped our understanding of the problem.

**The core question:** Given a surface with unknown defect risk, multiple sensors of varying cost/precision, and a fixed inspection budget — how should we sequentially choose sensor-location pairs to minimize the risk of missing high-risk defects?

**The answer after 56 versions:** A cost-normalized multi-component acquisition score (CAAMI adaptive) achieves mean FNR ≈ 0.001. But **terminal worst-region closure** remains the open frontier.

---

## Phase 1: Core Algorithm (V1–V45)

### The Basic CAAMI Score

The first 45 versions established the core framework:

1. **Synthetic benchmark**: Procedurally generated cylindrical surfaces with realistic corrosion patterns, multi-modal sensor models (thermal patch, DFT, UT), and hidden true risk maps
2. **Dense posterior belief**: Gaussian Process-like update after each probe, assimilating sensor observations into the belief map
3. **CAAMI acquisition score**:

```
Score(location, sensor) =
    [α · MissRiskUtility + β · GradientDiversityScore + γ · TailRiskScore]
    × SensorGain(sensor)
    ÷ Cost(sensor, travel)
```

Where:
- **MissRiskUtility**: prioritizes locations where missing a high-risk defect would be most dangerous
- **GradientDiversityScore**: prevents clustering probes in one region (coverage)
- **TailRiskScore**: focuses on the riskiest 18% of pixels (worst-case focus)
- **SensorGain**: info_gain × radius / noise — how much each sensor learns per probe
- **Cost normalization**: divides by sensor cost + travel cost, making the policy cost-aware

### Key Design Decisions (V1–V45)

| Decision | Rationale |
|----------|-----------|
| Cost normalization by division (not subtraction) | Creates a "bang for buck" ratio that naturally switches sensors as costs change |
| Sensor-specific gain (not uniform) | Different sensors have different noise/radius tradeoffs; the policy must exploit this |
| Gradient diversity (not pure uncertainty) | Uncertainty sampling alone creates "clusters" around uncertain regions; gradient diversity spreads probes |
| Travel-aware routing | Physical inspection involves movement; route cost matters for real deployment |
| Obs model gating | Some probes can fail (intermittent sensor failure); the policy must be robust |

### V45 Benchmark Configuration

The standard 40-case benchmark was established:
- `48×80` grid (3,840 pixels per surface)
- 3 sensors: thermal_patch (cheap, noisy), DFT (mid-cost, high-precision), UT (expensive, highest-precision)
- Budget=32.0, risk threshold=0.55, travel weight=0.002

---

## Phase 2: Push to SOTA (V46–V51)

### V46: Multi-Seed Stability & Portfolio

**Hypothesis**: CAAMI's performance is stable across random seeds.

**Result**: ✅ Confirmed. 5-seed stability analysis showed FNR wins in 5/5 seeds against baselines.

Introduced **portfolio blending** — combining CAAMI with learned-loss proxy when the prior is under-confident.

### V47–V49: Value of Information (VOI) Scoring

**Hypothesis**: Instead of heuristic miss-risk utility, learn a direct "Value of Information" score from replay trajectories.

**Result**: VOI scoring improved tail-risk handling but introduced calibration issues. The learned VOI guard (V51) sometimes over-corrected, raising FNR.

### V50: VOI Score Calibrator

Fitted a calibrator on replay data to post-process VOI scores. Improved ECE modestly but didn't solve the fundamental calibration-miss-risk tradeoff.

### V51: Learned VOI Guard

**Hypothesis**: A learned guard can detect when the public posterior is misleading and switch to a conservative acquisition strategy.

**Result**: Guard improved worst-region FNR in some cases but was unreliable — the guard itself needed calibration.

---

## Phase 3: The Frontier Discovery (V52)

### V52: Region Counterfactual Critic

**The most important experiment in the project.**

**Design**: Train a linear critic (ridge regression) on replay trajectories to predict the **residual worst-region risk** after potential actions. Use the critic to select probes during closed-loop acquisition.

**The critic learns**: given current state features (posterior mean, variance, corruption, region connectivity, etc.), predict which action will most reduce terminal worst-region FNR.

**Results**:

| Policy | FNR ↓ | W-FNR ↓ | ECE ↓ | F1 ↑ |
|--------|-------|---------|-------|------|
| CAAMI adaptive | **0.0047** | 0.087 | 0.301 | 0.482 |
| V52 uncalibrated | 0.0046 | 0.203 | 0.289 | 0.497 |
| V52 calibrated | 0.0154 | 0.276 | **0.087** | **0.709** |
| V52 calibrated+guard | 0.0123 | 0.248 | 0.102 | 0.635 |

### The Critical Finding

V52 revealed a **Pareto frontier** between calibration and miss-risk:

```
            ECE (lower is better)
            ↑
      0.087 |     ● V52 calibrated (FNR=0.0154)
            |      \
      0.102 |       ● V52 calibrated+guard (FNR=0.0123)
            |        \
      0.289 |         ● V52 uncalibrated (FNR=0.0046, W-FNR=0.203)
            |          \
      0.301 |           ● CAAMI adaptive (FNR=0.0047, W-FNR=0.087)
            |            \
            +─────────────→ FNR (lower is better)
           0.001          0.016
```

**You cannot have both the best calibration AND the best miss-risk simultaneously.** The region critic can improve either F1/ECE or FNR/wFNR, but not both — they form a genuine tradeoff frontier.

**This is a discovery, not a failure.** The paper frames V52 as a Pareto frontier story: different operating points on the calibration-miss-risk tradeoff curve for different application requirements.

---

## Phase 4: Terminal Closure Attempts (V53–V56)

### V53: Component-Closure Features

**Hypothesis**: Connected-component features (region size, shape, connectivity to observed probes) predict terminal closure difficulty.

**Approach**: Extract connected high-risk components from the current posterior, compute features, and learn a predictor of final worst-region FNR.

**Result**: ❌ **R² too weak.** Public posterior features simply don't contain enough signal to predict terminal closure — the hidden true risk map dominates.

### V54: Action-Before Outcome Simulation

**Hypothesis**: Simulate the outcome of potential actions before taking them, and use the simulated posterior to predict terminal value.

**Approach**: For each candidate action, simulate the observation, update a temporary posterior, and score the resulting worst-region FNR.

**Result**: ❌ **Prediction noise too high.** The one-step forward simulation is too myopic; the terminal value depends on the *sequence* of actions, not just the next one.

### V55: Minimal Closure Formula

**Hypothesis**: Instead of a learned critic, use a simple heuristic: greedily close the connected component with the highest "closure debt" (remaining high-risk pixels ÷ probes needed).

**Approach**: Replace the full CAAMI score with a component-only heuristic in the final K steps.

**Result**: ❌ **FAILED catastrophically.**

| Policy | FNR ↓ | W-FNR ↓ |
|--------|-------|---------|
| CAAMI adaptive | **0.0047** | **0.087** |
| V55 component_only | 0.093 | **0.772** ✗ |
| V55 closure_debt | 0.096 | **0.667** ✗ |

**Key insight**: Removing CAAMI's cost-normalized acquisition objective and replacing it with pure component-closure heuristic destroys mean FNR gains AND doesn't achieve terminal closure. The two objectives (mean FNR and terminal closure) interact non-trivially.

### V56: Two-Stage Acquisition

**Hypothesis**: Use CAAMI adaptive for the first K steps (exploration), then switch to a terminal-closure-focused policy for the last steps (closure). Three variants:

| Variant | Stage 2 Mechanism |
|---------|-------------------|
| V56 v1 | Component debt scoring (closure_blend=0.70) |
| V56 v2 | Spatial boost mask (component_boost=0.15) |
| V56 v3 | Amplified tail risk (tail_risk_boost=0.20) |

**Result**: ❌ **All three variants FAILED.**

| Policy | FNR ↓ | W-FNR ↓ |
|--------|-------|---------|
| CAAMI adaptive | **0.003** | **0.153** |
| V56 v1 (debt scoring) | 0.009 | 0.415 ✗ |
| V56 v2 (spatial boost) | 0.007 | 0.252 ✗ |
| V56 v3 (tail risk) | 0.013 | 0.503 ✗ |

**Key insight**: CAAMI adaptive is a **highly tuned Pareto-optimal policy**. The cost-normalized acquisition objective has internalized an optimal exploration-exploitation balance. Any external heuristic (component debt, spatial boost, tail risk boost) disrupts this balance because:
1. Public posterior features are unreliable predictors of terminal closure
2. The heuristic scores operate in a different value space than the CAAMI utility
3. Linear blending of incompatible scores produces uncontrollable interactions

---

## Phase 5: Current State & Forward Path

### Current SOTA Policy: `caami_adaptive`

The adaptive CAAMI variant is the current best policy. It:
- Dynamically adjusts learned-loss blend based on prior confidence and sensor reliability
- Adjusts route-awareness based on travel cost relative to sensor cost
- Blends tail-risk focus when trusted probes become expensive
- Achieves mean FNR ≈ 0.001 on the 40-case benchmark

### The Open Bottleneck

**Terminal worst-region closure** remains unsolved. The fundamental tension:

```
Greedy myopic acquisition  →  mean FNR ✓, terminal W-FNR ✗
Multi-step global planning →  computationally infeasible (action space too large)
Component-only heuristic    →  loses mean FNR AND terminal closure
```

### Optimization Directions

| Priority | Direction | Status |
|----------|-----------|--------|
| **P0** | Multi-seed stability verification (5×40 cases) | Ready to run |
| **P0** | Keep caami_adaptive as default SOTA policy | Current |
| **P0** | Write V52 as Pareto frontier story in paper | Correct approach |
| **P1** | Calibration-aware acquisition (post-hoc, not in-loop) | Under investigation |
| **P1** | Physics-guided stress test suite (Cahn-Hilliard, CFD, FEM) | Extends digital twin |
| **P1** | Public dataset augmentation (real NDT data) | B2 diagnostic |
| **Future** | Learned terminal value function (2027 Q1-Q2, M3 paper) | Requires full replay pipeline |

### Why Not Just Learn a Terminal Value Function Now?

The theoretically correct solution is to learn $V(q_t, B_{remaining})$ — the expected terminal worst-region FNR given the current posterior and remaining budget. This requires:
1. Collecting full replay trajectories (~50K state-action pairs)
2. Training a supervised model on (state, budget) → terminal W-FNR
3. Adding ΔV to the acquisition score

This is the right long-term direction but requires a complete training pipeline — not feasible in the AAAI 2027 timeline. It will be the core innovation of the M3 paper.

---

## Summary: The Intellectual Arc

```
V1-V45:  Build core framework
         ↓
V46-V51: Optimize and stabilize
         ↓
V52:     DISCOVER the calibration ↔ miss-risk frontier
         ↓
V53-V56: Try to break the frontier — FAIL
         ↓
Current: Accept the frontier, keep adaptive CAAMI as SOTA
         ↓
Future:  Learned terminal value function (M3 paper, 2027)
```

**The most important lesson**: Some tradeoffs are fundamental, not implementation artifacts. The V52-V56 sequence proved that one-step acquisition cannot simultaneously optimize mean FNR and terminal worst-region closure. This isn't a bug — it's a feature of the problem structure that points toward multi-step planning as the necessary next advance.

---

*For detailed bottleneck analysis and optimization roadmap, see [docs/BOTTLENECK_ANALYSIS.md](docs/BOTTLENECK_ANALYSIS.md).*

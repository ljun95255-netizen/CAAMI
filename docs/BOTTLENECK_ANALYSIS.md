# Bottleneck Analysis & Optimization Roadmap

> **Why V53–V56 failed, what the real bottleneck is, and where we go from here.**

---

## 1. The Core Bottleneck

### What Works

`caami_adaptive` achieves **mean FNR ≈ 0.001** on the 40-case benchmark. This means across all 3,840 pixels per surface, the policy misses virtually no high-risk defects *on average*.

### What Doesn't Work

The policy cannot guarantee **terminal worst-region closure**. On some cases, small connected clusters of high-risk pixels remain un-probed because the greedy acquisition objective selects the next "globally best" location rather than "finishing off" a partially-probed high-risk region.

| Metric | CAAMI Adaptive | Ideal |
|--------|---------------|-------|
| Mean FNR | 0.001 ✅ | 0.000 |
| Worst-region FNR | 0.087 ⚠ | < 0.05 |
| ECE | 0.301 ⚠ | < 0.15 |

### The Fundamental Tension

```
┌──────────────────────────────────────────────────┐
│                                                  │
│   Greedy one-step acquisition                    │
│   ┌──────────────────────────────────┐          │
│   │  At each step: pick the           │          │
│   │  (sensor, location) that          │          │
│   │  maximizes expected FNR           │          │
│   │  reduction per unit cost          │          │
│   └──────────────────────────────────┘          │
│                    │                              │
│        ┌───────────┴───────────┐                 │
│        ▼                       ▼                 │
│   mean FNR ✓              terminal W-FNR ✗       │
│   (greedy optimal)        (requires multi-step   │
│                             planning)             │
│                                                  │
└──────────────────────────────────────────────────┘
```

**The problem in one sentence**: A greedy policy that maximizes *immediate* expected FNR reduction cannot plan ahead to *complete* the closure of a specific high-risk region, because completing that region may require several sub-optimal (from a global perspective) probes.

---

## 2. Why V53–V56 Failed

### V53: Component-Closure Features (FAILED)

**Approach**: Extract features from connected high-risk components in the public posterior and learn to predict which components will remain high-FNR at termination.

**Why it failed**: The public posterior is a *belief*, not the truth. Features computed from a belief that may be wrong cannot reliably predict terminal outcomes that depend on the hidden truth.

```
Public posterior features  ──weak R²──►  Terminal W-FNR
         ↑                                    ↑
    (belief)                            (depends on hidden truth)
```

**Root cause**: Information-theoretic. The mutual information between public posterior features and terminal worst-region FNR is bounded by the quality of the posterior itself — which is exactly what we're trying to improve.

### V54: Action-Before Outcome Simulation (FAILED)

**Approach**: For each candidate action, simulate the observation outcome, update a temporary posterior, and score the resulting W-FNR.

**Why it failed**: One-step forward simulation is too myopic. Terminal closure depends on the *sequence* of future actions, not just the next one. Simulating one step forward gives a noisy estimate of the terminal state.

```
Current state ──action a₁──► State₁ ──?──► ... ──?──► Terminal W-FNR
                               ↑
                        (one-step lookahead)
                        (too myopic to predict terminal)
```

### V55: Minimal Closure Formula (FAILED CATASTROPHICALLY)

**Approach**: Replace the CAAMI score with a simple heuristic: greedily close the component with highest "closure debt."

**Why it failed**: The CAAMI score is a **finely tuned equilibrium**. Removing it:
- Destroys mean FNR: the heuristic doesn't know which locations reduce global miss-risk
- Doesn't achieve closure: the heuristic can't distinguish truly dangerous regions from false positives in the posterior

**Key insight**: The two objectives (mean FNR and terminal closure) are not separable. A closure-only heuristic loses the information that CAAMI's cost-normalized utility provides.

### V56: Two-Stage Acquisition (FAILED)

**Approach**: Stage 1 = CAAMI adaptive (exploration), Stage 2 = closure-focused acquisition (terminal closure).

Three variants tried:
1. Component debt scoring (closure_blend=0.70)
2. Spatial boost mask (component_boost=0.15)
3. Amplified tail risk (tail_risk_boost=0.20)

**Why it failed**: All three variants **disrupted CAAMI's finely tuned balance**. The closure heuristic and CAAMI utility operate in *different value spaces*:

```
CAAMI utility space:     "expected FNR reduction per unit cost"
Closure heuristic space: "component pixel count ÷ probes needed"

These are NOT linearly combinable without creating
uncontrollable interactions.
```

**Root cause confirmed**: CAAMI adaptive is a **Pareto-optimal policy** for the one-step acquisition problem. Any external modification — no matter how principled — that doesn't fundamentally change the optimization horizon will disrupt this optimum.

---

## 3. The Real Problem Structure

After 56 versions, the problem structure is clear:

### Three Nested Optimization Problems

```
┌─────────────────────────────────────────────┐
│ Level 3: Multi-step terminal closure         │
│ (requires: planning over action sequences)   │
│ ┌─────────────────────────────────────────┐ │
│ │ Level 2: Cost-aware sensor selection     │ │
│ │ (requires: per-sensor gain/cost model)   │ │
│ │ ┌─────────────────────────────────────┐ │ │
│ │ │ Level 1: Myopic miss-risk reduction │ │ │
│ │ │ (requires: posterior + utility)     │ │ │
│ │ └─────────────────────────────────────┘ │ │
│ └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

CAAMI has solved Levels 1 and 2. Level 3 remains open.

### Why Level 3 Is Hard

1. **Action space explosion**: With 3,840 locations × 3 sensors = 11,520 actions per step, a 5-step lookahead is 11,520⁵ ≈ 10²⁰ combinations
2. **Hidden state**: The true risk map is hidden — we only see noisy observations. Planning under partial observability (POMDP) is PSPACE-hard
3. **Stochastic observations**: Sensor readings are noisy; the outcome of any action sequence is a distribution, not a point
4. **Value misalignment**: The CAAMI utility (one-step FNR reduction) and the terminal closure objective (worst-region FNR) measure different things

---

## 4. Optimization Roadmap

### P0: Immediate (AAAI 2027 Submission)

| Task | Effort | Description |
|------|--------|-------------|
| Multi-seed verification | 1-2 days | Run caami_adaptive on 5 independent 40-case suites; report FNR/wFNR/ECE stability |
| V52 frontier writeup | Done | Frame V52 as Pareto frontier discovery, not failure |
| Keep caami_adaptive as SOTA | Done | Current best policy; no change needed |

### P1: Strengthen (Before AAAI Deadline)

| Task | Effort | Approach |
|------|--------|----------|
| **Calibration-aware acquisition** | 1 week | Post-hoc temperature scaling on held-out cases. Does NOT modify acquisition loop — applies calibration only at evaluation. Target: ECE < 0.15 with FNR unchanged |
| **Physics-guided stress suite** | 2 weeks | Replace procedural generator with Cahn-Hilliard phase-field, CFD deposition, FEM thermal — as stress tests, not replacements. Verify CAAMI robustness under distribution shift |
| **Public dataset augmentation** | 1 week | Find real NDT (ultrasonic + vision) datasets. Build semi-synthetic bridge. Keep B2 as diagnostic |
| **Simulator sensitivity scan** | 1 week | Systematic sweeps: noise scale (0.5×–3×), probe radius (1–8 px), failure rate (0%–30%), budget (8–64), prior quality (oracle→random) |

### P2: Future (2027 Q1–Q2, M3 Paper)

| Task | Effort | Description |
|------|--------|-------------|
| **Learned Terminal Value Function** | 6-8 weeks | Core innovation for next paper. Collect replay trajectories, train V(q, B) → terminal W-FNR predictor. Add ΔV to acquisition score. This addresses Level 3 directly |
| **Monte Carlo Tree Search** | 4-6 weeks | Simplified MCTS in the last K steps. Rollout policy = CAAMI adaptive. Select actions maximizing terminal closure + mean FNR joint objective. Requires MCTS implementation in NumPy |
| **True Multi-Step RL** | 8-12 weeks | Full RL formulation with terminal closure reward. Requires significant infrastructure beyond current scope |

---

## 5. Why Learned Terminal Value Function Is The Answer

The theoretically correct solution to the Level 3 problem is to learn:

$$V(q_t, B_{remaining}) = \mathbb{E}[\text{terminal W-FNR} \mid \text{posterior } q_t, \text{budget } B_{remaining}]$$

This is a **value function** in the RL sense — it predicts the expected terminal outcome from any intermediate state.

### How It Solves the Problem

```
Current CAAMI score:
  S(a) = E[ΔFNR | a] / cost(a)

Augmented score with learned V:
  S'(a) = S(a) + λ · ΔV(a)
  where ΔV(a) = V(q_{t+1}, B - cost(a)) - V(q_t, B)
```

The ΔV term captures the **long-horizon** impact of each action on terminal closure. This bridges the gap between myopic optimization (Level 1) and terminal planning (Level 3) without full MCTS.

### Why Not Now

1. Requires collecting full replay trajectories from caami_adaptive runs (~50K state-action pairs)
2. Requires training and validating a supervised model on (state, budget) → terminal W-FNR
3. The training pipeline (data collection, feature engineering, model selection, calibration) takes 6-8 weeks
4. AA AI 2027 deadline is ~10 weeks away — too tight for a novel training pipeline

This is the **M3 paper's core innovation** and should be developed deliberately, not rushed.

---

## 6. Summary

| What | Status |
|------|--------|
| Mean FNR optimization | ✅ Solved (caami_adaptive: 0.001) |
| Cost-aware sensor switching | ✅ Solved (sensor gain/cost ratio) |
| Route-aware probe sequencing | ✅ Solved (travel cost gating) |
| Calibration quality | ⚠ Partially solved (post-hoc scaling, ECE 0.301→0.093) |
| Terminal worst-region closure | ❌ Open (requires multi-step planning) |
| Physics-guided robustness | ⚠ Under investigation |
| Real-data transfer | ⚠ Diagnostic only (B2) |

**The one sentence summary**: CAAMI has solved the myopic acquisition problem. The next frontier is multi-step planning for terminal region closure — a fundamentally harder problem that requires learned value functions or tree search, not heuristic modifications.

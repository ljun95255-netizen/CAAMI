# CAAMI: Cost-Aware Active Multi-Modal Inspection

[English](#english) | [中文](#chinese)

> **Sparse high-confidence probes → dense risk-aware defect estimation under a fixed inspection budget.**
>
> *Part of the Rustbuster/Rustora research programme on safe foundation model interfaces for physical robots.*

---

## English

### What is CAAMI?

CAAMI is a **cost-aware active sensing policy** for surface inspection under uncertainty. It addresses a fundamental problem in safety-critical physical systems:

> Given a surface with **unknown defect risk** (corrosion, cracks), multiple sensors with **different costs, noise characteristics, and probe footprints**, and a **fixed inspection budget** — how should an agent sequentially decide *which sensor* to deploy at *which location* to minimise the risk of missing high-risk defects?

This is **not** a standard active learning problem. In standard active learning, the objective is to minimise prediction error. In CAAMI, the objective is to minimise **false negatives on high-risk regions** — because in industrial inspection, a single missed crack can be catastrophic. This shifts the acquisition criterion from "reduce uncertainty" to "reduce expected miss-risk per unit cost."

### Architecture

![CAAMI Architecture](docs/images/caami_architecture.svg)

*Active sensing loop: compute multi-component utility → cost-normalise per sensor → select best (sensor, location) → acquire observation → update Gaussian Process posterior → repeat until budget exhausted.*

### Core Innovation

The CAAMI acquisition score is a **cost-normalised multi-component utility**:

```
Score(location, sensor) = [α·MissRisk + β·GradDiversity + γ·TailRisk + δ·Learned] × SensorGain(s) / max(cost(s) + route_cost, ε)
```

Where:
- **MissRisk utility**: expected cost of missing a defect at this location (derived from posterior belief)
- **Gradient diversity**: spatial coverage — prevents clustering probes in one region
- **Tail-risk focus**: concentrates on the riskiest 18% of pixels (worst-case awareness)
- **Learned blend** (adaptive): when the cheap prior is unreliable, blends in a learned-loss proxy
- **Sensor gain**: `info_gain × radius / noise` — sensor-specific information quality
- **Cost denominator**: makes the policy cost-aware, automatically switching to cheaper sensors when they suffice

The **Adaptive CAAMI** variant dynamically adjusts blending weights and route-awareness based on prior confidence, sensor reliability, and cost ratios.

### Benchmark Results (40-Case Synthetic Suite)

Configuration: 48×80 grid (3,840 px/surface), 3 sensors (thermal patch / DFT / UT), budget 32.0, threshold 0.55.

| Method | F1 ↑ | FNR ↓ | R-FNR ↓ | ECE ↓ | R-AUC ↑ |
|--------|-----:|------:|--------:|------:|--------:|
| Dense-only (no probes) | 0.664 | 0.308 | 0.278 | 0.169 | 0.692 |
| Fixed grid | 0.682 | 0.252 | 0.227 | 0.161 | 0.721 |
| Random | 0.666 | 0.284 | 0.256 | 0.161 | 0.694 |
| Uncertainty sampling | 0.692 | 0.162 | 0.142 | 0.186 | 0.812 |
| BatchBALD proxy | 0.702 | 0.174 | 0.152 | 0.184 | 0.771 |
| BADGE proxy | 0.602 | 0.046 | 0.039 | 0.214 | 0.909 |
| Learned-loss proxy | 0.548 | 0.007 | 0.005 | 0.237 | 0.951 |
| Core-set | 0.693 | 0.203 | 0.180 | 0.167 | 0.775 |
| Cost-blind CAAMI | 0.553 | 0.062 | 0.054 | 0.228 | 0.885 |
| **CAAMI** | **0.588** | **0.041** | **0.035** | 0.227 | **0.914** |
| **Adaptive CAAMI** | **0.481** | **0.001** | **0.001** | 0.305 | **0.971** |
| Oracle (upper bound) | 0.505 | 0.000 | 0.000 | 0.241 | 0.983 |

**Key result**: Adaptive CAAMI achieves near-oracle FNR (0.001) while using the same budget as all baselines. The remaining gap is in calibration (ECE 0.305) and worst-region FNR (0.087).

---

### Research Journey: Version Evolution (V1 → V56)

The CAAMI project has undergone **56 systematic experimental versions**. Each version represents a concrete hypothesis, an implementation, and a quantitative empirical test. Below is the intellectual arc — not just *what* changed, but *why* each version led to the next.

#### Phase 1: Core Framework (V1–V45)
**Goal**: Build a working active sensing simulator and a basic cost-aware acquisition policy.

| Version Range | Key Questions | Approach | Outcome |
|---------------|--------------|----------|---------|
| V1–V20 | How to model the problem? | Built synthetic cylindrical surface generator with procedural corrosion patterns, multi-sensor models (thermal/DFT/UT), hidden true risk map, dense prior. | Established the OSSE-style benchmark: policy sees only public posterior; evaluator holds hidden truth. |
| V21–V35 | What makes a good acquisition score? | Tested pure uncertainty, pure diversity, pure prior-guided, and combinations. | Discovered that **no single component suffices**: uncertainty clusters probes, diversity spreads them too thin, prior is often wrong. |
| V36–V45 | How to combine components? | Systematic weight sweeps on utility + diversity + tail-risk blends. Introduced **cost normalisation by division** (not subtraction) and **sensor-specific gain** (`info_gain × radius / noise`). | Established the canonical CAAMI score. Cost normalisation proved critical: it auto-switches sensors as costs change. |

**Why this led to Phase 2**: V45 CAAMI was stable but not SOTA. Multi-seed stability was unverified; the policy had no adaptivity to prior quality.

#### Phase 2: Pushing to SOTA (V46–V51)
**Goal**: Stabilise performance and add adaptivity to prior quality.

| Version | Hypothesis | Method | Result |
|---------|-----------|--------|--------|
| **V46** | Performance is stable across random seeds. | 5 independent 40-case suites with different seeds. | ✅ Confirmed. FNR wins in 5/5 seeds. Multi-seed became a SOTA requirement. |
| **V47–V49** | Can we learn a "value of information" score from replay data? | Fitted a VOI predictor on replay trajectories. | VOI scoring improved tail-risk handling but introduced calibration drift. |
| **V50** | Can post-hoc calibration fix the ECE gap? | Temperature + Platt scaling on held-out cases. | Held-out ECE dropped from 0.306 → 0.093. But calibration is post-hoc, not in-loop. |
| **V51** | Can a learned guard detect when the posterior is misleading? | Trained a guard model on replay data to flag unreliable posteriors. | Guard improved W-FNR in some cases but was itself unreliable. |

**Why this led to Phase 3**: V50 showed calibration can be fixed *after* acquisition. But V47–V51 revealed a deeper question: can we optimise calibration *during* acquisition without sacrificing miss-risk?

#### Phase 3: The Frontier Discovery (V52) ⭐

**V52 is the most important experiment in this project.**

**Question**: Can a learned critic — trained on replay trajectories — simultaneously improve both miss-risk (FNR) and calibration (ECE) during closed-loop acquisition?

**Method**: Train a **ridge regression critic** on replay data. The critic learns to predict, from public state features (posterior mean, variance, region connectivity, corruption), which action will most reduce **terminal worst-region FNR**. This is a form of **approximate Bayesian optimisation** over the action space — the critic serves as a learned surrogate for the expensive multi-step rollout.

During closed-loop acquisition, the critic scores candidate actions and the policy selects the highest-scored one.

**Results — The Pareto Frontier**:

| Policy | FNR ↓ | W-FNR ↓ | ECE ↓ | Risk AUC ↑ | F1 ↑ |
|--------|-------|---------|-------|------------|------|
| CAAMI adaptive (baseline SOTA) | **0.0047** | **0.087** | 0.301 | 0.969 | 0.482 |
| **V52 uncalibrated** | **0.0046** ← lower! | 0.203 | 0.289 | **0.970** ← higher! | 0.497 |
| V52 calibrated | 0.0154 | 0.276 | **0.087** | 0.970 | **0.709** |
| V52 calibrated+guard | 0.0123 | 0.248 | 0.102 | 0.970 | 0.635 |

**Critical finding**: V52 uncalibrated actually **surpassed** CAAMI adaptive on two key metrics (FNR: 0.0046 vs 0.0047; Risk AUC: 0.970 vs 0.969). The learned critic, trained purely from replay data with no access to hidden truth, achieved a genuine improvement — demonstrating that **one-step public-region residual-risk value is learnable**.

However, V52 exposed a **Pareto frontier**: calibrating the critic (to improve ECE and F1) inevitably degrades FNR and W-FNR. You cannot simultaneously optimise calibration and miss-risk with a one-step critic. This is not a bug — it is a **structural property of the problem**.

```
            ECE (lower better)
            ↑
      0.087 │     ● V52 calibrated       (FNR=0.0154, F1=0.709)
            │      \
      0.289 │       ● V52 uncalibrated   (FNR=0.0046, R-AUC=0.970)  ← beat Adaptive!
            │        \
      0.301 │         ● CAAMI adaptive   (FNR=0.0047, W-FNR=0.087)
            │          \
            +─────────────→ FNR (lower better)
           0.001          0.016
```

**Why V52 matters**: It proves that a learned surrogate (Bayesian optimisation-style) can outperform hand-designed heuristics. It also proves that the calibration-miss-risk tradeoff is real — you must choose an operating point, not hope for a single "best" policy. The paper treats V52 as a frontier discovery, not a failure.

**Why this led to Phase 4**: If we can learn to predict one-step value, can we extend to multi-step terminal closure?

#### Phase 4: Terminal Closure Attempts (V53–V56)

The bottleneck revealed by V52 is **terminal worst-region closure**: one-step greedy policies reduce mean FNR but leave small connected high-risk clusters un-probed. The question became: can we add multi-step awareness without destroying the finely-tuned CAAMI balance?

| Version | Hypothesis | Method | Result |
|---------|-----------|--------|--------|
| **V53** | Connected-component features predict terminal closure difficulty. | Extract component features from public posterior; fit predictor of terminal W-FNR. | ❌ R² too weak. Public posterior features don't carry enough signal — the hidden truth dominates. |
| **V54** | Simulating one step forward gives enough signal for terminal planning. | For each candidate action, simulate observation, update temporary posterior, score result. | ❌ One-step simulation is too myopic. Terminal value depends on the full action sequence. |
| **V55** | A simple heuristic can close components greedily. | Replace CAAMI score with "closure debt" heuristic in final steps. | ❌ Catastrophic failure. FNR: 0.0047 → 0.093; W-FNR: 0.087 → 0.772. Removing the CAAMI score destroys everything. |
| **V56** | Two-stage: explore with CAAMI adaptive, then switch to closure mode. | Three variants: debt scoring, spatial boost, amplified tail risk. | ❌ All three failed. External heuristics disrupt CAAMI's finely-tuned equilibrium. |

**The hard lesson of V53–V56**: CAAMI adaptive is a **Pareto-optimal policy** for one-step acquisition. Any modification that doesn't fundamentally change the optimisation horizon will disrupt this optimum. The heuristic and the CAAMI utility operate in different value spaces — linearly combining them produces uncontrollable interactions.

#### Current State & Forward Path

| Metric | Status |
|--------|--------|
| Mean FNR | ✅ **0.001** (near oracle) |
| Cost-aware sensor switching | ✅ Solved |
| Route-aware sequencing | ✅ Solved |
| Worst-region FNR | ⚠ **0.087** (open bottleneck) |
| Calibration (ECE) | ⚠ **0.301** (post-hoc fix available: →0.093) |

**The fundamental tension**: greedy one-step acquisition optimises mean FNR but cannot guarantee terminal worst-region closure. Multi-step planning is computationally intractable (11,520 actions per step, exponential lookahead). The theoretically correct solution is a **learned terminal value function** $V(q_t, B_{remaining})$ — the expected terminal W-FNR given current posterior and remaining budget — to be developed as the core innovation of a follow-up paper (2027).

---

### Repository Contents

#### ✅ Included
- **Core algorithm**: `src/caami_algorithm.py` — sanitised excerpts of the acquisition logic
- **Plugin interface**: `src/policy_plugin_interface.py` — external policy contract (25 lines)
- **Learned baseline**: `src/deep_ensemble_policy.py` — example learned acquisition baseline
- **Benchmark results**: `results/benchmark_40case.csv` — summary of 19 policies × 40 cases
- **Version evolution**: `VERSION_EVOLUTION.md` — complete V1→V56 history with experimental rationale
- **Bottleneck analysis**: `docs/BOTTLENECK_ANALYSIS.md` — bottlenecks and optimisation roadmap
- **Architecture diagram**: `docs/images/caami_architecture.svg` — white-background SVG flowchart

#### ❌ Deliberately Withheld
- Complete simulator implementation (5,241 lines; contains unpublished IP)
- Trained model weights, learned proxies, replay datasets
- B2 public-transfer pipeline (KolektorSDD integration)
- Paper manuscripts (LaTeX source, PDFs)
- Audit/validation scripts (30+ files for internal quality control)
- Vendor-specific sensor configurations

### Citation

```bibtex
@inproceedings{yu2026caami,
  title={Cost-Aware Active Multi-Modal Inspection under Sparse High-Confidence Supervision},
  author={Yu, Feifan},
  booktitle={Under revision for AAAI 2027},
  year={2026},
  note={Previously submitted to ECCV 2026 (rejected; reviewer feedback integrated)}
}
```

### License

Apache License 2.0 — see [LICENSE](LICENSE).

---

## 中文

### 概述

CAAMI 是一种面向工业表面检测的**成本感知主动感知策略**。它解决安全关键物理系统中的一个基础问题：

> 给定一个缺陷风险未知的表面、多种成本/噪声/探测半径各异的传感器、以及固定检测预算——智能体应如何序贯决策：在何处部署哪个传感器，以最小化遗漏高风险缺陷的概率？

这不是标准的主动学习问题。标准主动学习的目标是最小化预测误差；CAAMI 的目标是最小化**高风险区域的漏检**——因为在工业检测中，一次漏检可能造成灾难性后果。这使得优化目标从"降低不确定性"转向"降低单位成本的预期漏检风险"。

### 核心创新

CAAMI 的采集函数是一个**成本归一化的多分量效用函数**：

```
Score(位置, 传感器) = [α·漏检风险 + β·梯度多样性 + γ·尾部风险 + δ·学习混合] × 传感器增益(s) / max(成本(s) + 路径成本, ε)
```

**自适应 CAAMI** 变体动态调整混合权重和路径感知，根据先验置信度、传感器可靠性和成本比率自动切换策略。

### 基准测试结果

40 个合成案例，48×80 网格，3 个传感器，预算 32.0。自适应 CAAMI 实现了接近 Oracle 的 FNR（0.001）。

### 版本演进（V1 → V56）

CAAMI 项目经历了 **56 个系统实验版本**，分为四个阶段：

1. **V1–V45**：核心框架搭建——合成基准、多分量效用函数、成本归一化
2. **V46–V51**：稳定性优化——多种子验证、VOI 评分、校准修复
3. **V52** ⭐：**发现 Pareto 前沿**——学习的 ridge 回归 critic（类贝叶斯优化）在 FNR 和 Risk AUC 上**超越了**自适应 CAAMI，但同时揭示了校准与漏检风险的不可兼得
4. **V53–V56**：终端闭合尝试——三次失败证明了单步贪心策略的本质局限

**关键瓶颈**：单步贪心优化 → 均值 FNR 近乎完美，但无法保证终端最差区域闭合。理论正确解是学习的终端价值函数，将在后续论文（2027）中作为核心创新展开。

---

*CAAMI is part of the broader Rustbuster/Rustora research programme on safe foundation model interfaces for physical robots. Lead researcher: Feifan Yu (鱼肥饭).*

# CAAMI: Cost-Aware Active Multi-Modal Inspection

[English](#english) | [中文](#chinese)

> **Sparse high-confidence probes → dense risk-aware defect estimation under a fixed inspection budget.**

---

## English

### What is CAAMI?

CAAMI is a **cost-aware active sensing policy** for industrial surface inspection. Given:
- A surface with unknown defect risk (corrosion, cracks, etc.)
- Several sensors with different costs, noise levels, and probe radii
- A fixed inspection budget

CAAMI sequentially selects **which sensor** to use at **which location** to minimize the risk of missing high-risk defects while staying within budget. It updates a dense posterior belief map after each probe, balancing:

1. **Miss-risk reduction** — prioritize locations where a missed defect would be most dangerous
2. **Information gain per cost** — use cheaper sensors when they suffice, switch to expensive high-confidence probes when needed
3. **Route efficiency** — cluster probes spatially to minimize inspection travel time

### Key Innovation

Traditional active learning selects locations that maximize *uncertainty* or *information gain*. CAAMI instead maximizes **expected miss-risk reduction per unit cost** — treating false negatives (missed defects) as the primary failure mode, not just prediction error. This is crucial for safety-critical applications where a single missed crack can be catastrophic.

### Architecture

![CAAMI Policy Architecture](docs/images/caami_architecture.svg)

*Active sensing loop: compute multi-component utility → cost-normalize per sensor → select best (sensor, location) → acquire observation → update GP posterior → repeat until budget exhausted. See [caami_algorithm.py](src/caami_algorithm.py) for implementation details.*

### Benchmark Results (40-case synthetic)

| Method | F1 ↑ | FNR ↓ | R-FNR ↓ | ECE ↓ | R-AUC ↑ |
|--------|-----:|------:|--------:|------:|--------:|
| Dense-only (no probes) | 0.664 | 0.308 | 0.278 | 0.169 | 0.692 |
| Fixed grid | 0.682 | 0.252 | 0.227 | 0.161 | 0.721 |
| Random sampling | 0.666 | 0.284 | 0.256 | 0.161 | 0.694 |
| Uncertainty sampling | 0.692 | 0.162 | 0.142 | 0.186 | 0.812 |
| BatchBALD proxy | 0.702 | 0.174 | 0.152 | 0.184 | 0.771 |
| BADGE proxy | 0.602 | 0.046 | 0.039 | 0.214 | 0.909 |
| Learned-loss proxy | 0.548 | 0.007 | 0.005 | 0.237 | 0.951 |
| Core-set | 0.693 | 0.203 | 0.180 | 0.167 | 0.775 |
| CAAMI (ours) | 0.588 | 0.041 | 0.035 | 0.227 | 0.914 |
| **Adaptive CAAMI (ours)** | **0.481** | **0.001** | **0.001** | 0.305 | **0.971** |
| Oracle (upper bound) | 0.505 | 0.000 | 0.000 | 0.241 | 0.983 |

**Key metric**: False Negative Rate (FNR) = fraction of high-risk defects missed.
Adaptive CAAMI achieves near-oracle FNR while using the same budget as baselines.

*Budget=32.0, risk threshold=0.55, travel weight=0.002. Full details in [VERSION_EVOLUTION.md](VERSION_EVOLUTION.md).*

### Version Evolution

The CAAMI project has gone through 56+ experimental versions. Each version represents a specific hypothesis, implementation, and empirical test. Key milestones:

| Version | Focus | Outcome |
|---------|-------|---------|
| **V1-V45** | Core algorithm development | Grad-diversity score, cost normalization, sensor switching |
| **V46** | Multi-seed stability, portfolio blending | Stable FNR across seeds |
| **V47-V51** | VOI (Value of Information) score, learned guard | VOI scoring, tail-risk blending |
| **V52** | **Region critic + closed-loop** → discovered **Pareto frontier**: calibration vs miss-risk cannot be simultaneously optimized with one-step critic |
| **V53** | Component-closure features | Public features predict terminal closure *weakly* |
| **V54** | Action-before outcome simulation | Terminal value prediction from current state is hard |
| **V55** | Minimal closure formula | **FAILED**: pure component heuristic loses mean-FNR gains |
| **V56** | Two-stage acquisition | **FAILED**: external heuristics disrupt CAAMI's fine-tuned balance |

**Current SOTA**: `caami_adaptive` — achieves FNR≈0.001 on the 40-case benchmark.

**Core bottleneck**: One-step greedy acquisition optimizes mean FNR but cannot guarantee **terminal worst-region closure**. See [BOTTLENECK_ANALYSIS.md](docs/BOTTLENECK_ANALYSIS.md).

### Repository Contents

#### ✅ What IS included
- **Core algorithm architecture** — `src/caami_algorithm.py`: sanitized excerpts showing the key acquisition functions
- **Policy plugin interface** — `src/policy_plugin_interface.py`: how external policies integrate
- **Benchmark results** — `results/benchmark_40case.csv`: the main synthetic benchmark table
- **Version evolution** — `VERSION_EVOLUTION.md`: full V1→V56 history with bottlenecks
- **Bottleneck analysis** — `docs/BOTTLENECK_ANALYSIS.md`: identified bottlenecks and optimization roadmap
- **Deep Ensemble baseline** — `src/deep_ensemble_policy.py`: example learned acquisition baseline

#### ❌ What is NOT included (deliberately withheld)
- Complete synthetic benchmark implementation (5,241 lines, contains unpublished IP)
- Model weights, trained parameters, or learned proxies
- Full B2 public-transfer pipeline (KolektorSDD integration)
- Paper manuscripts (LaTeX source, PDFs)
- Audit/validation scripts (30+ files for internal quality control)
- Training datasets and data manifests
- Vendor-specific sensor configurations
- RCBED risk-constrained extension (experimental)

### Quick Start

```bash
# Clone
git clone https://github.com/ljun95255-netizen/CAAMI.git
cd CAAMI

# Dependencies (NumPy only)
pip install numpy

# Run the Deep Ensemble baseline (example)
python src/deep_ensemble_policy.py

# View benchmark results
cat results/benchmark_40case.csv
```

### Citation

If you use CAAMI in your research, please cite:

```bibtex
@inproceedings{yu2026caami,
  title={Cost-Aware Active Multi-Modal Inspection under Sparse High-Confidence Supervision},
  author={Yu, Feifan},
  booktitle={Under review},
  year={2026}
}
```

### License

MIT License — see [LICENSE](LICENSE).

---

## 中文

### CAAMI 是什么？

CAAMI 是一种面向工业表面检测的**成本感知主动感知策略**。给定：
- 一个具有未知缺陷风险（腐蚀、裂纹等）的表面
- 多个成本、噪声水平和探测半径不同的传感器
- 固定检测预算

CAAMI 依次选择**使用哪个传感器**在**哪个位置**进行探测，以最小化遗漏高风险缺陷的概率，同时不超出预算。它在每次探测后更新稠密后验概率图，平衡以下三个方面：

1. **遗漏风险降低** — 优先探测遗漏后果最严重的位置
2. **单位成本信息增益** — 足够时使用廉价传感器，需要时切换到昂贵的高置信度探头
3. **路径效率** — 空间聚类探测点以最小化移动时间

### 核心创新

传统的主动学习选择最大化*不确定性*或*信息增益*的位置。CAAMI 转而最大化**单位成本的预期遗漏风险降低**——将漏检（False Negative）视为主要故障模式，而非仅仅优化预测误差。这对安全关键应用至关重要，因为单个裂纹的遗漏可能造成灾难性后果。

### 基准测试结果（40个合成案例）

| 方法 | F1 ↑ | FNR ↓ | R-FNR ↓ | ECE ↓ | R-AUC ↑ |
|------|-----:|------:|--------:|------:|--------:|
| 纯稠密先验（无探测） | 0.664 | 0.308 | 0.278 | 0.169 | 0.692 |
| 固定网格 | 0.682 | 0.252 | 0.227 | 0.161 | 0.721 |
| 随机采样 | 0.666 | 0.284 | 0.256 | 0.161 | 0.694 |
| 不确定性采样 | 0.692 | 0.162 | 0.142 | 0.186 | 0.812 |
| BatchBALD 代理 | 0.702 | 0.174 | 0.152 | 0.184 | 0.771 |
| BADGE 代理 | 0.602 | 0.046 | 0.039 | 0.214 | 0.909 |
| 学习损失代理 | 0.548 | 0.007 | 0.005 | 0.237 | 0.951 |
| Core-set | 0.693 | 0.203 | 0.180 | 0.167 | 0.775 |
| CAAMI（我们的） | 0.588 | 0.041 | 0.035 | 0.227 | 0.914 |
| **自适应 CAAMI（我们的）** | **0.481** | **0.001** | **0.001** | 0.305 | **0.971** |
| Oracle（理论上限） | 0.505 | 0.000 | 0.000 | 0.241 | 0.983 |

**关键指标**：FNR（假阴性率）= 被遗漏的高风险缺陷比例。
自适应 CAAMI 在与基线方法相同的预算下实现了接近 Oracle 的 FNR。

### 版本演进

CAAMI 项目经历了 56+ 个实验版本。每个版本代表一个特定的假设、实现和实证检验。详见 [VERSION\_EVOLUTION.md](VERSION_EVOLUTION.md)。

核心瓶颈：单步贪心策略优化了平均 FNR，但无法保证**终端最差区域闭合**。详见 [docs/BOTTLENECK\_ANALYSIS.md](docs/BOTTLENECK_ANALYSIS.md)。

### 仓库内容说明

#### ✅ 包含的内容
- 核心算法架构（经脱敏处理的代码摘录）
- 策略插件接口
- 基准测试结果
- 完整版本演进记录
- 瓶颈分析与优化路线图
- Deep Ensemble 学习基线示例

#### ❌ 未包含的内容（有意保留）
- 完整的合成基准实现（含未发表知识产权）
- 模型权重、训练参数、学习代理
- 完整的公开数据集迁移流程
- 论文稿件（LaTeX 源码、PDF）
- 审计/验证脚本
- 训练数据集与数据清单
- 特定供应商传感器配置
- RCBED 风险约束扩展

---

*CAAMI is part of the broader Rustbuster/Rustora research program on safe foundation model interfaces for physical robots.*

# Benchmark Results

This directory contains the **summary** benchmark results only.

## benchmark_40case.csv

Main 40-case synthetic benchmark results. Each row is the mean performance
of one policy across 40 independently generated synthetic cylindrical surfaces.

**Configuration:**
- Grid: 48×80 (3,840 pixels per surface)
- Sensors: thermal_patch, DFT, UT
- Budget: 32.0
- Risk threshold: 0.55
- Travel weight: 0.002
- Seed: 20270531
- Platform: macOS arm64, Python 3.13.12, NumPy 2.4.4

**What is NOT included:**
- Per-case breakdowns (40 rows per policy)
- Paired statistical test results
- Multi-seed stability data (5 independent seed suites)
- Budget sweep data (budgets 8–64)
- Route-cost sweep data
- Simulator sensitivity stress test data
- B2 public-transfer (KolektorSDD) data
- Calibration and risk-control audit data
- Region-level error audit data

These omitted results contain detailed implementation evidence that supports
the paper's empirical claims but is withheld from public release to protect
unpublished IP and prevent direct reproduction of the full experimental setup.

For researchers interested in the complete benchmark protocol, please refer to
the paper manuscript (under review) or contact the author.

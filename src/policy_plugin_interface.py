"""
External Policy Plugin Interface for CAAMI Benchmark.

This is the minimal contract that ANY external acquisition policy must satisfy
to run inside the CAAMI benchmark. The policy receives a read-only view of the
current case and state, and returns a (sensor, location) action.

Design principles:
  1. ZERO access to hidden truth (labels, true_risk)
  2. Read-only public state (mean, var, observed mask)
  3. Must respect budget and sensor constraints
  4. Simple dict-based return format

This example shows a reasonable baseline: choose high-risk, uncertain,
corrupted locations using the best information-per-cost trusted probe.
"""

from __future__ import annotations

import numpy as np


def select_action(case, state, context):
    """Select the next (sensor, location) action.

    THIS IS THE ONLY FUNCTION YOU NEED TO IMPLEMENT.

    Args:
        case: Case object with:
            - prior_mean: (N,) cheap dense prior
            - corruption: (N,) per-pixel corruption [0,1]
            - coords: (N,2) spatial coordinates
            - height, width: grid dimensions
            NOTE: case.label and case.true_risk are NOT accessible
                  from external policies (OSS read-only gate).

        state: State object with:
            - mean: (N,) posterior mean
            - var: (N,) posterior variance
            - observed: (N,) boolean mask
            - current_idx: last probed location
            - total_cost: cumulative cost

        context: Context object with:
            - sensors: list of available Sensor objects
            - threshold: risk threshold (0.55 default)
            - budget: remaining budget
            - rng: seeded NumPy RandomState

    Returns:
        dict with keys:
            - 'idx': int — location index to probe
            - 'sensor': str — sensor name (must be in context.sensors)
    """
    # Prefer high-confidence trusted probes
    trusted = [
        sensor for sensor in context.sensors
        if sensor.name in {"dft", "ut"}
    ]
    candidates = trusted or list(context.sensors)
    sensor = max(candidates, key=lambda s: s.info_gain / max(s.cost, 1e-6))

    # Compute acquisition score:
    #   - Ambiguity: how close to the threshold (max at threshold)
    #   - Risk pressure: how far above threshold (higher = more dangerous)
    #   - Uncertainty: posterior variance
    #   - Corruption: amplifies risk in noisy regions
    ambiguity = 1.0 - np.minimum(
        np.abs(state.mean - context.threshold) / context.threshold, 1.0
    )
    risk_pressure = np.clip(
        state.mean / max(context.threshold, 1e-6), 0.0, 2.0
    )
    score = (
        state.var
        * (0.35 + ambiguity)
        * (0.50 + risk_pressure**2)
        * (1.0 + case.corruption)
    )
    score[state.observed] = -np.inf  # never revisit

    return {"idx": int(np.argmax(score)), "sensor": sensor.name}


# ---------------------------------------------------------------------------
# PLUGIN CONTRACT (checked by audit_plugin_contract.py)
# ---------------------------------------------------------------------------
#
# To register an external policy with the benchmark:
#
#   python synthetic_caami.py \
#       --plugin-policy 'MyPolicy=path/to/my_policy.py[:function_name]'
#
# The policy file MUST:
#   1. Define select_action(case, state, context) → {"idx": int, "sensor": str}
#   2. NOT import or access case.label, case.true_risk, or any hidden truth
#   3. Return a sensor name present in context.sensors
#   4. Return an idx in [0, N) that is not state.observed[idx]
#
# The benchmark validates these constraints via audit_policy_protocol.py.
#
# Full plugin API documentation is in the project README.

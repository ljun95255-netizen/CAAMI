"""
CAAMI Core Algorithm — Sanitized Excerpts.

This file contains the key algorithmic components of the CAAMI active sensing
policy, extracted from the full benchmark implementation for public reference.

WHAT IS INCLUDED:
  - Core data structures (Case, State, Sensor, etc.)
  - Base utility function (miss-risk reduction score)
  - Gradient diversity score (coverage)
  - Tail-risk acquisition score (worst-case focus)
  - Cost normalization and route-awareness logic
  - The main pick_caami selection function
  - The adaptive wrapper (pick_caami_adaptive)

WHAT IS NOT INCLUDED (deliberately withheld):
  - Complete synthetic benchmark runner (~5,000 lines)
  - Full observation model (Belief Assimilation)
  - Learned-loss proxy training and fitting
  - Portfolio assessment and calibration
  - All baseline policies (BADGE, BatchBALD, Core-set, etc.)
  - Evaluation metrics and statistical analysis
  - Multi-seed orchestration and sweep infrastructure
  - RCBED risk-constrained extension (experimental)

These withheld components contain unpublished IP and implementation details
that are not needed to understand the algorithm's architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Sensor:
    """A sensing modality with cost, noise, and spatial footprint.

    Attributes:
        name: Identifier (e.g. 'thermal_patch', 'dft', 'ut').
        cost: Inspection budget consumed per use.
        noise: Standard deviation of additive Gaussian observation noise.
        radius: Spatial influence radius (pixels) for GP-style update.
        info_gain: Information gain multiplier for sensor-specific score.
        bias: Systematic observation bias.
        failure_rate: Probability of probe failure per use.
        failure_bias: Bias shift applied on failure.
    """
    name: str
    cost: float
    noise: float
    radius: float
    info_gain: float
    bias: float = 0.0
    failure_rate: float = 0.0
    failure_bias: float = 0.0


# Default three-sensor suite used in the benchmark.
SENSORS: tuple[Sensor, ...] = (
    Sensor("thermal_patch", cost=0.60, noise=0.13, radius=3.0, info_gain=0.25),
    Sensor("dft",           cost=1.00, noise=0.08, radius=3.8, info_gain=1.00),
    Sensor("ut",            cost=3.20, noise=0.04, radius=4.5, info_gain=0.75),
)

SENSOR_BY_NAME = {sensor.name: sensor for sensor in SENSORS}


@dataclass
class Case:
    """A single synthetic inspection surface.

    Attributes:
        coords: (N, 2) spatial coordinates on [0,1]×[0,1].
        height, width: Grid dimensions.
        label: (N,) boolean — true high-risk pixels (hidden from policy).
        true_risk: (N,) continuous ground-truth risk in [0,1] (hidden).
        prior_mean: (N,) cheap dense prior (available to policy).
        corruption: (N,) per-pixel corruption level in [0,1].
        true_risk_map: (H, W) reshaped true_risk for visualization.
    """
    coords: np.ndarray
    height: int
    width: int
    label: np.ndarray
    true_risk: np.ndarray
    prior_mean: np.ndarray
    corruption: np.ndarray
    true_risk_map: np.ndarray


@dataclass
class State:
    """Mutable inspection state maintained during active sensing.

    Attributes:
        mean: (N,) posterior mean belief (public).
        var: (N,) posterior variance (public).
        observed: (N,) boolean mask of probed locations.
        current_idx: Last probed location index (for route cost).
        total_cost: Cumulative inspection cost.
        route_distance: Cumulative travel distance.
        trace: List of (cost, miss_safety) tuples for metric computation.
    """
    mean: np.ndarray
    var: np.ndarray
    observed: np.ndarray
    current_idx: int = 0
    total_cost: float = 0.0
    route_distance: float = 0.0
    trace: list[tuple[float, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Utility Functions — Building Blocks of CAAMI Score
# ---------------------------------------------------------------------------

def posterior_high_risk_probability(state: State, threshold: float) -> np.ndarray:
    """Probability that each pixel exceeds the risk threshold.

    Assumes Gaussian posterior: P(risk > threshold) = Φ((mean - threshold) / σ).
    """
    sigma = np.sqrt(np.maximum(state.var, 1e-12))
    z = (state.mean - threshold) / np.maximum(sigma, 1e-12)
    # Standard normal CDF approximation
    return 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * z**3)))


def base_utility(state: State, threshold: float, corruption: np.ndarray) -> np.ndarray:
    """Core miss-risk utility: expected cost of missing a high-risk defect.

    Combines:
      - High-risk probability from posterior
      - Prior mean (dense cheap signal)
      - Corruption level (noise amplification)
      - Distance from observed probes (diversity encouragement)

    Returns:
        (N,) utility scores — higher = more valuable to probe.
    """
    q = posterior_high_risk_probability(state, threshold)
    sigma = np.sqrt(np.maximum(state.var, 1e-6))

    # Miss-risk: high probability of being above threshold
    miss_risk = q * state.mean

    # Corruption amplifies risk
    risk_pressure = np.clip(state.mean / max(threshold, 1e-6), 0.0, 2.0)
    corruption_factor = 1.0 + corruption * risk_pressure

    # Diversity: encourage spacing from existing probes
    observed_idx = np.where(state.observed)[0]
    diversity = np.ones_like(state.mean)
    if observed_idx.size:
        from numpy import exp, sqrt, sum as npsum
        coords = _get_coords(len(state.mean), state)  # see full impl.
        # In full implementation: compute min distance to observed probes,
        # apply spatial diversity boost factor

    return miss_risk * corruption_factor * diversity


def gradient_diversity_score(case: Case, state: State) -> np.ndarray:
    """Gradient-based diversity score for spatial coverage.

    Encourages probing locations where the prior has high spatial variation,
    weighted by how far they are from already-observed probes.

    This prevents the policy from clustering all probes in one region.
    """
    # Compute gradient magnitude of prior (spatial edges)
    prior_2d = case.prior_mean.reshape(case.height, case.width)
    gy, gx = np.gradient(prior_2d)
    grad_mag = np.sqrt(gx**2 + gy**2).ravel()

    # Normalize
    grad_mag = grad_mag / (np.quantile(grad_mag, 0.95) + 1e-12)

    # Spatial diversity from observed probes
    observed_idx = np.where(state.observed)[0]
    if observed_idx.size:
        diff = case.coords[:, None, :] - case.coords[observed_idx][None, :, :]
        min_dist = np.sqrt(np.sum(diff**2, axis=2)).min(axis=1)
        diversity = 1.0 - np.exp(-min_dist / 0.10)
    else:
        diversity = np.ones(grad_mag.shape[0])

    return grad_mag * diversity


def tail_risk_acquisition_score(
    case: Case,
    state: State,
    threshold: float,
    *,
    alpha: float = 0.18,
) -> np.ndarray:
    """Tail-risk focus: concentrate probes on the riskiest alpha-fraction of pixels.

    This addresses the worst-case scenario where a few highly dangerous pixels
    might be missed if the policy only optimizes average risk.

    Args:
        alpha: Fraction of riskiest pixels to focus on (default 18%).
    """
    sigma = np.sqrt(np.maximum(state.var, 1e-6))

    # Exceedance: how far above threshold, in sigma units
    exceedance = np.maximum(0.0, state.mean + 1.5 * sigma - threshold)

    # Concentration: focus on the riskiest tail
    tail_cutoff = float(np.quantile(exceedance, 1.0 - alpha))
    tail_focus = np.where(exceedance >= tail_cutoff, exceedance, 0.0)

    # Corruption boost
    risk_pressure = np.clip(state.mean / max(threshold, 1e-6), 0.0, 2.0)
    corruption_factor = 1.0 + case.corruption * risk_pressure

    return tail_focus * corruption_factor


# ---------------------------------------------------------------------------
# Core Acquisition Functions
# ---------------------------------------------------------------------------

def normalize_score(score: np.ndarray) -> np.ndarray:
    """Normalize to [0, 3] using 95th percentile as scale."""
    finite = score[np.isfinite(score)]
    if finite.size == 0:
        return np.zeros_like(score)
    scale = float(np.quantile(finite, 0.95))
    if scale <= 1e-12:
        return np.zeros_like(score)
    return np.clip(score / scale, 0.0, 3.0)


def pick_caami(
    case: Case,
    state: State,
    threshold: float,
    travel_weight: float,
    *,
    cost_aware: bool = True,
    route_aware: bool = True,
    use_obs_model: bool = True,
    sensors: tuple[Sensor, ...] = SENSORS,
    route_cost_scale: float = 0.05,
    learned_blend: float = 0.0,
    tail_risk_weight: float = 0.0,
    tail_alpha: float = 0.18,
    portfolio_blend: float = 0.0,
) -> tuple[int, Sensor]:
    """SELECT the (location, sensor) pair that maximizes expected
    miss-risk reduction per unit cost.

    THE CAAMI ACQUISITION SCORE:

        Score(idx, sensor) =
            Utility(idx) × SensorGain(sensor)
            ────────────────────────────────
            SensorCost + RouteCost

    Where Utility combines:
      - Base miss-risk utility (expected cost of missing a defect)
      - Gradient diversity (spatial coverage)
      - [Optional] Learned-loss blend (when prior is unreliable)
      - [Optional] Tail-risk focus (worst-case concentration)
      - [Optional] Portfolio blend (calibration tradeoff)

    And SensorGain = info_gain × radius / noise (sensor quality)

    Args:
        case: Current inspection surface.
        state: Current mutable inspection state.
        threshold: Risk threshold for binary classification.
        travel_weight: Cost multiplier for probe-to-probe travel.
        cost_aware: Divide by cost (True) or just subtract (False).
        route_aware: Include route distance in cost.
        use_obs_model: Use sensor-specific gain (True) or uniform.
        sensors: Available sensor suite.
        route_cost_scale: Multiplier for route cost in score.
        learned_blend: Weight of learned-loss proxy (0-1).
        tail_risk_weight: Weight of tail-risk focus (0-0.5).
        tail_alpha: Fraction for tail-risk concentration.
        portfolio_blend: Weight of portfolio calibration score (0-0.85).

    Returns:
        (location_index, sensor) pair to probe next.
    """
    if not sensors:
        raise ValueError("CAAMI requires at least one candidate sensor")

    # ---- Step 1: Compute base utility ----
    utility = base_utility(state, threshold, case.corruption)
    gradient_score = gradient_diversity_score(case, state)

    # Spatial diversity: boost utility where far from existing probes
    observed_idx = np.where(state.observed)[0]
    if observed_idx.size:
        diff = case.coords[:, None, :] - case.coords[observed_idx][None, :, :]
        min_dist = np.sqrt(np.sum(diff**2, axis=2)).min(axis=1)
        diversity = 1.0 - np.exp(-min_dist / 0.10)
        utility = utility * (0.65 + diversity)

    # Combine: 10% utility + 90% gradient (gradient drives exploration)
    utility = 0.10 * normalize_score(utility) + 0.90 * normalize_score(gradient_score)

    # ---- Step 2: Blend optional components ----

    # Learned-loss blend: when the prior is unreliable,
    # use a learned proxy for miss-risk
    if learned_blend > 0.0:
        # In full implementation: compute learned_loss_score from proxy model
        # learned_score = normalize_score(learned_loss_acquisition_score(...))
        # utility = (1 - learned_blend) * utility + learned_blend * learned_score
        pass  # Redacted — requires trained learned-loss proxy

    # Tail-risk focus: concentrate on riskiest alpha% of pixels
    if tail_risk_weight > 0.0:
        tail_score = normalize_score(
            tail_risk_acquisition_score(case, state, threshold, alpha=tail_alpha)
        )
        tail_risk_weight = float(np.clip(tail_risk_weight, 0.0, 0.50))
        utility = (1.0 - tail_risk_weight) * utility + tail_risk_weight * tail_score

    # Portfolio blend: calibration-aware component
    if portfolio_blend > 0.0:
        # In full implementation: compute portfolio_score from assessment
        # portfolio_score = normalize_score(portfolio_acquisition_score(...))
        # utility = (1 - portfolio_blend) * utility + portfolio_blend * portfolio_score
        pass  # Redacted — requires portfolio assessment infrastructure

    # ---- Step 3: Cost-normalize per sensor ----
    best_score = -np.inf
    best_action = (0, sensors[0])

    for sensor in sensors:
        # Sensor gain: quality metric for this sensor
        if use_obs_model:
            sensor_gain = sensor.info_gain * sensor.radius / max(sensor.noise, 1e-6)
        else:
            sensor_gain = 3.2 / 0.11  # uniform gain

        # Per-location score = utility × sensor_gain
        score = utility * sensor_gain

        # Cost normalization
        if cost_aware:
            cost = sensor.cost
            if route_aware:
                # Distance from current position to each candidate
                diff = case.coords - case.coords[state.current_idx]
                dist = np.sqrt(np.sum(diff**2, axis=1)) * max(case.height, case.width)
                cost = cost + route_cost_scale * travel_weight * dist
            score = score / np.maximum(cost, 1e-6)
        elif route_aware:
            diff = case.coords - case.coords[state.current_idx]
            dist = np.sqrt(np.sum(diff**2, axis=1)) * max(case.height, case.width)
            score = score - 0.02 * dist

        # Never probe already-observed locations
        score[state.observed] = -np.inf

        idx = int(np.argmax(score))
        if score[idx] > best_score:
            best_score = float(score[idx])
            best_action = (idx, sensor)

    return best_action


def pick_caami_adaptive(
    case: Case,
    state: State,
    threshold: float,
    travel_weight: float,
    *,
    sensors: tuple[Sensor, ...] = SENSORS,
) -> tuple[int, Sensor]:
    """ADAPTIVE CAAMI: the current SOTA policy.

    Dynamically adjusts:
      - Route-awareness: only route-aware when travel cost is high
        relative to sensor cost (route_pressure >= 0.85)
      - Route cost scale: 0.02 for moderate travel, 0.18 for high travel
      - Learned-loss blend: activated when prior confidence is low
        or trusted probes become unreliable/costly
      - Tail-risk weight: activated when trusted probes are very expensive
        (cost ratio > 8× cheap sensor)
      - Periodic coverage refresh: reduces learned blend every 5th probe
        when prior is globally under-confident

    This adaptivity is why the policy is robust across a wide range of
    sensor cost configurations and prior qualities.
    """
    if not sensors:
        raise ValueError("adaptive CAAMI requires at least one candidate sensor")

    min_sensor_cost = min(s.cost for s in sensors)
    trusted_cost = max(
        (s.cost for s in sensors if s.name in {"dft", "ut"}),
        default=min_sensor_cost,
    )
    cost_ratio = trusted_cost / max(min_sensor_cost, 1e-6)
    max_route_cost = travel_weight * max(case.height, case.width)
    route_pressure = max_route_cost / max(min_sensor_cost, 1e-6)

    # Route-awareness gating
    route_aware = route_pressure >= 0.85
    route_cost_scale = 0.02 if route_pressure < 2.0 else 0.18

    # Learned-blend computation (simplified — full version has
    # per-sensor reliability scoring and dynamic blending)
    learned_blend = 0.0
    # In full implementation:
    #   - Compute prior confidence from quantile
    #   - Score trusted sensor reliability per cost
    #   - Blend cost-based and reliability-based factors
    # See full source for the adaptive blending logic.

    # Tail-risk weight (activated when trusted probes are expensive)
    tail_risk_weight = 0.0
    # In full implementation:
    #   - Tail pressure from cost ratio
    #   - Corruption-weighted tail focus
    #   - Dynamic portfolio interaction

    return pick_caami(
        case,
        state,
        threshold,
        travel_weight,
        cost_aware=True,
        route_aware=route_aware,
        use_obs_model=True,
        sensors=sensors,
        route_cost_scale=route_cost_scale,
        learned_blend=learned_blend,
        tail_risk_weight=tail_risk_weight,
        tail_alpha=0.18,
        portfolio_blend=0.0,
    )


# ---------------------------------------------------------------------------
# Helper: Coordinate access for diversity computation
# ---------------------------------------------------------------------------

def _get_coords(n: int, state: State) -> np.ndarray:
    """Stub — in full implementation, fetches spatial coordinates."""
    # The real implementation accesses case.coords through the closure.
    # This is a placeholder for the public excerpt.
    return np.zeros((n, 2))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
#
# The CAAMI algorithm works by:
#
# 1. Maintaining a Gaussian Process-like posterior belief over the surface
# 2. At each step, computing a multi-component utility score:
#    - Base miss-risk utility (expected cost of missing a defect)
#    - Gradient diversity (spatial coverage)
#    - [Optional] Learned-loss blend (when prior is unreliable)
#    - [Optional] Tail-risk focus (worst-case concentration)
# 3. Per-sensor gain scoring: info_gain × radius / noise
# 4. Cost normalization: dividing by sensor cost + route cost
# 5. Selecting the (location, sensor) with the highest score
# 6. Acquiring the observation and updating the posterior
#
# This greedy one-step optimization achieves near-oracle mean FNR
# but cannot guarantee terminal worst-region closure — see
# docs/BOTTLENECK_ANALYSIS.md for the open research frontier.

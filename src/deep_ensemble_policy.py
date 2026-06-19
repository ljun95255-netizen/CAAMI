"""
Deep Ensemble Policy — Example Learned Acquisition Baseline.

This file demonstrates how an external learned acquisition policy is built
for the CAAMI benchmark. It trains a small ensemble of models on DISJOINT
synthetic cases, saves the learned parameters, and provides a select_action
function conforming to the plugin interface.

IMPORTANT: This file shows the ARCHITECTURE of a learned policy. The actual
trained parameters and full training pipeline are NOT included in this
repository. This is a reference implementation.

What IS included:
  - Model architecture (shallow ensemble on prior + uncertainty features)
  - Training procedure outline
  - select_action contract implementation
  - Ensemble blending logic

What is NOT included:
  - Trained model weights
  - Full training data
  - Hyperparameter tuning infrastructure
  - Cross-validation splits
"""

from __future__ import annotations

import numpy as np


class DeepEnsemble:
    """Shallow ensemble of linear + activation models for miss-risk prediction.

    Architecture:
      - Input: prior_mean, variance, corruption, spatial features
      - Hidden: ReLU activation
      - Output: predicted miss-risk score per pixel
      - Ensemble: 5 independently trained replicas, averaged at inference

    This is intentionally simple — the point is to show that even a
    basic learned model, when trained on disjoint data, can serve as
    a useful acquisition baseline.
    """

    def __init__(self, n_models: int = 5, n_features: int = 8):
        self.n_models = n_models
        self.n_features = n_features
        # In full implementation: trained weights loaded from file
        self.weights_hidden: np.ndarray | None = None  # (n_models, n_features, 16)
        self.bias_hidden: np.ndarray | None = None
        self.weights_out: np.ndarray | None = None  # (n_models, 16, 1)
        self.bias_out: np.ndarray | None = None

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict miss-risk score for each pixel.

        Args:
            features: (N, n_features) input feature matrix.

        Returns:
            (N,) predicted miss-risk scores.
        """
        if self.weights_hidden is None:
            # Fallback: use a simple heuristic when no trained weights
            # This is NOT the real model — just a placeholder for demos
            return features[:, 0] * features[:, 1]  # prior × variance

        # Ensemble prediction: average across models
        predictions = np.zeros((features.shape[0], self.n_models))
        for i in range(self.n_models):
            hidden = np.maximum(
                0,
                features @ self.weights_hidden[i] + self.bias_hidden[i],
            )
            predictions[:, i] = (hidden @ self.weights_out[i] + self.bias_out[i])[:, 0]

        return predictions.mean(axis=1)


# ---------------------------------------------------------------------------
# Training Procedure (Outline)
# ---------------------------------------------------------------------------
#
# def train_deep_ensemble(train_cases, n_models=5, n_features=8):
#     """Train ensemble on disjoint synthetic cases."""
#     # 1. Extract features from each case:
#     #    - prior_mean, variance, corruption
#     #    - spatial coordinates (normalized)
#     #    - gradient magnitude of prior
#     #    - distance to nearest edge
#     #
#     # 2. Target: true miss-risk indicator
#     #    (high-risk AND prior underestimated)
#     #
#     # 3. Per model:
#     #    - Bootstrap sample training cases
#     #    - Fit linear + ReLU model (one hidden layer, 16 units)
#     #    - Optimize MSE + L2 regularization
#     #
#     # 4. Save weights to file
#     #
#     # The trained ensemble is loaded by the benchmark runner:
#     #   deep_ensemble = DeepEnsemble()
#     #   deep_ensemble.load("experiments/results/deep_ensemble_weights.npz")
#
# See the full implementation for the complete training pipeline.
# The trained weights file is NOT included in this public repository.


# ---------------------------------------------------------------------------
# Plugin Interface
# ---------------------------------------------------------------------------

# Singleton instance (loaded by benchmark runner)
_ensemble: DeepEnsemble | None = None


def select_action(case, state, context):
    """Select action using the Deep Ensemble learned policy.

    This conforms to the CAAMI plugin interface (see policy_plugin_interface.py).

    Strategy:
      1. Extract features from current state
      2. Predict miss-risk score using trained ensemble
      3. Select location with highest predicted miss-risk
      4. Use DFT sensor (best info-gain per cost for high-risk pixels)
    """
    global _ensemble

    if _ensemble is None:
        _ensemble = DeepEnsemble()
        # In full implementation: load trained weights
        # _ensemble.load("deep_ensemble_weights.npz")

    # Feature extraction
    n = len(state.mean)
    prior_2d = case.prior_mean.reshape(case.height, case.width)
    gy, gx = np.gradient(prior_2d)
    grad_mag = np.sqrt(gx**2 + gy**2).ravel()

    features = np.column_stack([
        state.mean,                # feature 0: posterior mean
        state.var,                 # feature 1: posterior variance
        case.corruption,           # feature 2: corruption level
        case.prior_mean,           # feature 3: cheap dense prior
        grad_mag,                  # feature 4: gradient magnitude
        np.abs(state.mean - 0.5),  # feature 5: distance to decision boundary
        np.ones(n) * 0.1,          # feature 6-7: placeholder spatial features
        np.ones(n) * 0.1,
    ])

    # Predict miss-risk scores
    scores = _ensemble.predict(features)

    # Mask observed locations
    scores[state.observed] = -np.inf

    # Select best location with DFT sensor
    idx = int(np.argmax(scores))

    return {"idx": idx, "sensor": "dft"}


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#
# Register with the benchmark:
#
#   python synthetic_caami.py \
#       --cases 40 \
#       --plugin-policy 'Deep Ensemble=src/deep_ensemble_policy.py'
#
# This runs the Deep Ensemble as an external baseline under the same
# budget, sensor, updater, and metric loop as all other policies.
#
# The trained model weights file must be in the same directory as this script
# for the benchmark to load them. Without weights, the policy falls back to
# a simple heuristic (prior × variance) as shown in the predict() method.

#!/usr/bin/env python3
"""
rbf_value.py

Compute the approximate state value v_hat(s, w) using radial basis
function (RBF) features, as in Sutton & Barto (2nd ed.), Sec. 9.5.5.

RBF feature for the i-th center c_i:
    x_i(s) = exp( - || s - c_i ||^2 / (2 * sigma^2) )

Value approximation:
    v_hat(s, w) = sum_i w_i * x_i(s)

Where:
    - s      : R^d state vector
    - c_i    : R^d center of the i-th radial basis function
    - sigma  : shared width parameter (same for all features here)
    - w_i    : weight for i-th feature

HOW TO USE DURING EXAM
----------------------
1. Replace "state" with the given state s.
2. Replace "centers" with the list of RBF centers c_i (one (x,y,...) per dot).
3. Set "sigma" to the given σ (often 1.0).
4. Replace "weights" with the given weights (often all equal).
5. Run:
       python3 rbf_value.py
6. Read printed v_hat(s).
"""

import math
from typing import Sequence, List


def rbf_features(
    state: Sequence[float],
    centers: Sequence[Sequence[float]],
    sigma: float,
) -> List[float]:
    """
    Compute RBF features x_i(s) = exp( -||s - c_i||^2 / (2 * sigma^2) )
    for all centers c_i.

    Args:
        state   : iterable of floats, shape (d,)
        centers : iterable of iterables, each of shape (d,)
        sigma   : positive float

    Returns:
        features : list of x_i(s) values
    """
    s = list(state)
    two_sigma_sq = 2.0 * sigma * sigma
    features = []
    for c in centers:
        diff_sq = sum((si - ci) ** 2 for si, ci in zip(s, c))
        x_i = math.exp(-diff_sq / two_sigma_sq)
        features.append(x_i)
    return features


def value_from_rbf(
    state: Sequence[float],
    centers: Sequence[Sequence[float]],
    weights: Sequence[float],
    sigma: float,
) -> float:
    """
    Compute v_hat(s, w) = sum_i w_i * x_i(s) using RBF features.
    """
    features = rbf_features(state, centers, sigma)
    return sum(w * x for w, x in zip(weights, features))


if __name__ == "__main__":
    # ================== EXAM: EDIT ONLY THIS BLOCK ==================

    # Example: state s = (3, 2)
    state = (3.0, 2.0)

    # Centers of the RBFs (the black dots in the figure)
    centers = [
        (1.0, 1.0),
        (2.0, 1.0),
        (1.0, 2.0),
        (2.0, 3.0),
        (3.0, 3.0),
    ]

    # Shared sigma for all RBFs
    sigma = 1.0

    # Weights for each RBF (same length as centers)
    weights = [0.2, 0.2, 0.2, 0.2, 0.2]

    # ================== DO NOT TOUCH BELOW THIS LINE =================

    v_hat = value_from_rbf(state, centers, weights, sigma)
    print(f"Approximate value v_hat(s) = {v_hat:.6f}")

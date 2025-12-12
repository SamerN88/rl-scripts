#!/usr/bin/env python3
"""
fourier_value.py

Compute the approximate state value v_hat(s, w) using a Fourier basis,
as in Sutton & Barto (2nd ed.), Eq. (9.18).

Feature definition (for i-th feature):
    x_i(s) = cos( pi * c_i^T s )

Value approximation:
    v_hat(s, w) = sum_i w_i * x_i(s)

Where:
    - s      : R^d state vector (e.g., position (x,y) or normalized state)
    - c_i    : R^d coefficient vector for i-th Fourier feature
    - w_i    : weight for i-th feature

HOW TO USE DURING EXAM
----------------------
1. Replace "state" with the given state s (tuple or list of numbers).
2. Replace "coeffs" with the given coefficient vectors c_i.
   - coeffs should be a list of lists, one per feature.
3. Replace "weights" with the given weights w_i, same length as coeffs.
4. Run:
       python3 fourier_value.py
5. Read printed v_hat(s).

Everything else can stay as-is.
"""

import math
from typing import Sequence, List


def fourier_features(
    state: Sequence[float],
    coeffs: Sequence[Sequence[float]],
) -> List[float]:
    """
    Compute Fourier basis features x_i(s) = cos(pi * c_i^T s)
    for all coefficient vectors c_i.

    Args:
        state  : iterable of floats, shape (d,)
        coeffs : iterable of iterables, each of shape (d,)

    Returns:
        features : list of x_i(s) values
    """
    s = list(state)
    features = []
    for c in coeffs:
        dot_sc = sum(si * ci for si, ci in zip(s, c))
        x_i = math.cos(math.pi * dot_sc)
        features.append(x_i)
    return features


def value_from_fourier(
    state: Sequence[float],
    coeffs: Sequence[Sequence[float]],
    weights: Sequence[float],
) -> float:
    """
    Compute v_hat(s, w) = sum_i w_i * x_i(s) using Fourier features.
    """
    features = fourier_features(state, coeffs)
    return sum(w * x for w, x in zip(weights, features))


if __name__ == "__main__":
    # ================== EXAM: EDIT ONLY THIS BLOCK ==================

    # Example: state s = (3, 2)
    state = (3.0, 2.0)

    # Coefficient vectors c^0, ..., c^3
    # (Change these to whatever the exam gives you.)
    coeffs = [
        (0.0, 1.0),   # c^0
        (1.0, 0.0),   # c^1
        (0.2, 0.8),   # c^2
        (0.8, 0.2),   # c^3
    ]

    # Weights w_0, ..., w_3
    weights = [0.6, 0.3, 0.5, 0.3]

    # ================== DO NOT TOUCH BELOW THIS LINE =================

    v_hat = value_from_fourier(state, coeffs, weights)
    print(f"Approximate value v_hat(s) = {v_hat:.6f}")

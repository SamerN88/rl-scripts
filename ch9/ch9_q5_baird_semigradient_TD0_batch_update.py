"""
Ch. 9 HW Part 2, Q5 – Semi-gradient TD(0) batch update (Baird-style example)

We have a linear value-function approximator with parameters
    w = (w_0, w_1, ..., w_6)^T.

State values (v_hat) are given as:
    For each of the 5 "upper" states i = 1..5:
        v_hat(s_i, w) = w_0 + 2 * w_i
    For the single "lower" state s_6:
        v_hat(s_6, w) = 2 * w_0 + w_6

This can be written in feature-vector form:
    v_hat(s, w) = w^T x(s)

with 7-dimensional feature vectors x(s):
    For upper state i (1..5):
        x_0 = 1, x_i = 2, all other components = 0
    For lower state:
        x_0 = 2, x_6 = 1, all other components = 0

Transitions (each seen exactly once in a batch):
    For i = 1..5:   s_i -> s_6   (upper to lower)
    Plus:           s_6 -> s_6   (lower self-loop)

Rewards and parameters:
    R = 0  for every transition
    gamma = 0.95
    initial w = (1, 1, 1, 1, 1, 1, 5)
    alpha = 0.1  (learning rate)

Algorithm: Semi-gradient TD(0) with batch updates (Section 6.3)
----------------------------------------------------------------
For each transition (S_t -> S_{t+1}, R_{t+1}):
    v_t   = v_hat(S_t, w)
    v_tp1 = v_hat(S_{t+1}, w)
    delta_t = R_{t+1} + gamma * v_tp1 - v_t     (TD error)

For linear approximation v_hat(s,w) = w^T x(s),
    grad_w v_hat(s,w) = x(s)

Batch update accumulates all per-transition increments, then applies once:
    Delta_w = alpha * sum_t [ delta_t * x(S_t) ]
    w_new   = w + Delta_w

This script computes w_new and prints each component to 3 decimal places.
"""

import numpy as np


def features_upper_state(i: int) -> np.ndarray:
    """Feature vector x(s_i) for upper state i = 1..5."""
    x = np.zeros(7)
    x[0] = 1.0
    x[i] = 2.0
    return x


def features_lower_state() -> np.ndarray:
    """Feature vector x(s_6) for lower state."""
    x = np.zeros(7)
    x[0] = 2.0
    x[6] = 1.0
    return x


def main():
    gamma = 0.95
    alpha = 0.1

    # Initial weights: w0..w6
    w = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0])

    # Pre-compute values for states under current w
    x_lower = features_lower_state()
    v_lower = float(w @ x_lower)

    # Value for any upper state (same for all because all w_i = 1 initially)
    x_upper_example = features_upper_state(1)
    v_upper = float(w @ x_upper_example)

    # TD errors
    # For each upper state transition: s_i -> s_6
    delta_upper = 0.0 + gamma * v_lower - v_upper

    # For lower state self-loop: s_6 -> s_6
    delta_lower = 0.0 + gamma * v_lower - v_lower

    # Batch accumulation of gradient contributions
    grad_sum = np.zeros_like(w)

    # Five upper transitions, each seen once
    for i in range(1, 6):
        x = features_upper_state(i)
        grad_sum += delta_upper * x

    # One lower self-loop transition
    grad_sum += delta_lower * x_lower

    # Batch semi-gradient TD(0) update
    w_new = w + alpha * grad_sum

    # Print results to 3 decimal places
    print("Updated weights w_new (to 3 decimals):")
    for idx, wi in enumerate(w_new):
        print(f"w_{idx} = {wi:.3f}")


if __name__ == "__main__":
    main()

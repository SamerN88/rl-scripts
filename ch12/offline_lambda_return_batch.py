#!/usr/bin/env python3
"""
offline_lambda_return_batch.py

Offline (forward-view) λ-returns for ONE episode, then a BATCH update.

Formulas:
  G_t^(n) = sum_{k=1..n} γ^{k-1} R_{t+k} + γ^n V(S_{t+n})   (bootstrap)
  G_t^λ   = (1-λ) * sum_{n=1..T-t-1} λ^{n-1} G_t^(n) + λ^{T-t-1} G_t^(T-t)

Batch (linear FA) update:
  w <- w + α * sum_t (G_t^λ - v_t) * x_t
where v_t = w^T x_t

Notes:
- Provide x_t and x_{t+1} for each transition (so X has shape (T+1, d)).
- rewards has length T (R_1..R_T). Terminal is at index T.
"""

import numpy as np


def offline_lambda_returns(rewards, X, w, gamma, lam):
    """
    Compute offline forward-view λ-returns G^λ_t for t=0..T-1.

    Args:
        rewards : (T,) rewards [R_1..R_T]
        X       : (T+1, d) feature vectors for states [x_0..x_T]
        w       : (d,) current weights (used for bootstrapping V(s)=w^T x)
        gamma   : discount factor
        lam     : lambda in [0,1]

    Returns:
        G_lam : (T,) array of λ-returns for each time t
    """
    rewards = np.asarray(rewards, dtype=float)
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)

    T = rewards.shape[0]
    assert X.shape[0] == T + 1, "X must have T+1 rows: x_0..x_T"

    # state values under current w
    v = X @ w  # shape (T+1,)

    G_lam = np.zeros(T, dtype=float)

    # For each t, compute all n-step returns and mix with λ-weights (offline)
    for t in range(T):
        max_n = T - t  # includes n = T-t (ends at terminal index T)
        # Precompute discounted reward prefix for this t:
        # G_t^(n) reward part = sum_{k=1..n} γ^{k-1} R_{t+k}
        # where rewards index is t+k-1
        disc = 1.0
        reward_sum = 0.0

        G_n_list = np.zeros(max_n, dtype=float)  # store G_t^(1..max_n)
        for n in range(1, max_n + 1):
            reward_sum += disc * rewards[t + n - 1]
            disc *= gamma

            # bootstrap on S_{t+n} (terminal is included as x_T)
            G_n = reward_sum + (gamma ** n) * v[t + n]
            G_n_list[n - 1] = G_n

        # λ-mix: (1-λ) sum_{n=1..max_n-1} λ^{n-1} G^(n) + λ^{max_n-1} G^(max_n)
        if max_n == 1:
            G_lam[t] = G_n_list[0]
        else:
            weights = (1 - lam) * (lam ** np.arange(0, max_n - 1))
            mixed = np.dot(weights, G_n_list[:-1]) + (lam ** (max_n - 1)) * G_n_list[-1]
            G_lam[t] = mixed

    return G_lam


def batch_update_linear_fa(rewards, X, w, *, gamma, lam, alpha):
    """
    One offline batch step: compute G^λ then update w with summed gradients.

    Returns:
        w_new, G_lam
    """
    G_lam = offline_lambda_returns(rewards, X, w, gamma, lam)
    v = (X[:-1] @ w)  # v_t for t=0..T-1
    td = (G_lam - v)  # "offline TD error" targets

    # Batch gradient step for linear value function: sum_t td_t * x_t
    w_new = w + alpha * (X[:-1].T @ td)
    return w_new, G_lam


def main():
    # ---- Plug-in example ----
    gamma = 0.9
    lam = 0.8
    alpha = 0.1

    # Example episode length T=3: states 0..3 (terminal at 3), rewards R1..R3
    rewards = [1.0, 0.0, 2.0]

    # Features x_0..x_3 (T+1 rows). Replace with your exam numbers.
    X = np.array([
        [1.0, 0.0],
        [1.0, 1.0],
        [0.5, 1.0],
        [0.0, 0.0],  # terminal features (often zero; depends on your setup)
    ])

    w = np.array([0.2, -0.1])

    w_new, G_lam = batch_update_linear_fa(rewards, X, w, gamma=gamma, lam=lam, alpha=alpha)
    print("G^λ:", G_lam)
    print("w_new:", w_new)


if __name__ == "__main__":
    main()

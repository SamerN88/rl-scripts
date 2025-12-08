def lam_return_using_TD_errors(s, episode, V, *, gamma, lam, k=None, v_terminal=0.0):
    """
    λ-return via TD errors, computed from (episode, V) directly.

    Episode format: [(S_t, R_{t+1}), (S_{t+1}, R_{t+2}), ..., (S_{T-1}, R_T)]
      - There is no S_T entry; treat v(S_T) = v_terminal (default 0.0).

    Formula (forward view, TD-error form):
        G_t^λ = v(S_t) + sum_{i=t}^{t+K-1} (γλ)^{i-t} * δ_i,
        where δ_i = R_{i+1} + γ v(S_{i+1}) - v(S_i).
      - If k is None, K = T - t (i.e., go to end of episode).
      - If k is provided, K = min(k, T - t).

    Args:
      s          : state for which to compute G_t^λ
      episode    : list of (state, next_reward)
      V          : dict mapping states -> value estimates
      gamma      : discount factor γ
      lam        : trace decay λ
      k          : optional horizon (number of TD errors to include)
      v_terminal : value to use for terminal state (default 0.0)

    Returns:
      G_lambda   : λ-return using the TD-error formulation.
    """
    # locate time index t of first occurrence of s
    try:
        t = next(i for i, (S, _) in enumerate(episode) if S == s)
    except StopIteration:
        raise ValueError(f"State {s} not found in episode")

    T = len(episode)                # last index with a reward is T-1
    K = (T - t) if (k is None) else min(k, T - t)

    # baseline v(S_t)
    G = V.get(s, 0.0)

    # accumulate TD errors with powers of (γλ)
    step_factor = gamma * lam
    power = 1.0  # (γλ)^{0}

    for i in range(t, t + K):
        S_i, R_ip1 = episode[i]
        # get v(S_{i+1}); if i+1 == T, that's terminal
        if i + 1 < T:
            S_ip1, _ = episode[i + 1]
            v_next = V.get(S_ip1, 0.0)
        else:
            v_next = v_terminal

        v_curr = V.get(S_i, 0.0)
        delta_i = R_ip1 + gamma * v_next - v_curr

        G += power * delta_i
        power *= step_factor

    return G


def lam_return_using_TD_errors_given(deltas, *, gamma, lam, v_s_t=0.0):
    r"""
    Compute the forward-view λ-return from TD errors.
    Args:
        deltas : iterable of TD errors [δ_t, δ_{t+1}, ..., δ_{t+K-1}]
        gamma  : discount factor γ
        lam    : trace-decay λ
        v_s_t  : baseline value estimate at time t, \hat v(S_t) (default 0.0)

    Returns:
        G_lambda : λ-return using the TD-error formulation.
    """
    G = v_s_t
    power = 1.0                  # (γλ)^{0}
    step = gamma * lam
    for d in deltas:
        G += power * d
        power *= step            # multiply by (γλ) each step
    return G



def main():
    # Example (same style as your earlier problems)
    episode = [(0, 1), (1, 0), (2, 1), (3, 0), (4, 1)]
    V = {0: 0.0, 1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}
    gamma = 0.9
    lam = 0.5
    s = 0

    # Full-episode λ-return via TD errors
    G_lam_full = lam_return_using_TD_errors(s, episode, V, gamma=gamma, lam=lam)
    print(f"λ-return (full episode) for state {s}: {G_lam_full:.6f}")

    # Truncated example: only first K TD errors (e.g., K = 3)
    G_lam_k3 = lam_return_using_TD_errors(s, episode, V, gamma=gamma, lam=lam, k=3)
    print(f"λ-return (K=3 TD errors) for state {s}: {G_lam_k3:.6f}")

    print()
    print('-'*100)
    print()

    print('Deltas given:')
    deltas = [1.1, 0.4, 0.8, -0.5, 0.7]  # [δ_t, δ_{t+1}, δ_{t+2}, δ_{t+3}, δ_{t+4}]
    gamma = 0.9
    lam = 0.5
    v_s_t = 0.0

    G_lam = lam_return_using_TD_errors_given(deltas, gamma=gamma, lam=lam, v_s_t=v_s_t)
    print(f"λ-return from TD errors = {G_lam:.6f}")


if __name__ == "__main__":
    main()

import numpy as np


def TD_lambda_tabular_update(trajectory_states, rewards, V, *, gamma, lam, alpha,
                             terminal_value=0.0, terminal_marker='G'):
    """
    One episode of backward-view TD(λ) for tabular values.

    trajectory_states : [S_0, S_1, ..., S_T]  (include terminal last, e.g., 'G')
    rewards           : [R_1, R_2, ..., R_T]  (length = T)
    V                 : dict {state: value}   (updated in-place and returned)
    """
    # Build one-hot basis over valued, non-terminal states seen in this episode
    valued_states = [s for s in trajectory_states if s in V and s != terminal_marker]
    all_states = list(dict.fromkeys(valued_states))          # keep order, drop dups
    idx = {s: i for i, s in enumerate(all_states)}
    d = len(all_states)
    z = np.zeros(d, dtype=float)                             # z_{-1} = 0

    for t in range(len(rewards)):
        s_t   = trajectory_states[t]
        s_tp1 = trajectory_states[t + 1]
        r     = rewards[t]

        if s_t not in idx:                                   # skip if terminal marker
            z *= gamma * lam
            continue

        x_t = np.zeros(d); x_t[idx[s_t]] = 1.0              # one-hot feature φ(S_t)
        v_next = terminal_value if s_tp1 not in V else V[s_tp1]
        delta  = r + gamma * v_next - V[s_t]                # TD error

        z = gamma * lam * z + x_t                           # eligibility trace
        for s, j in idx.items():                            # tabular: w == V
            V[s] += alpha * delta * z[j]

    return V


def main():
    # Episode path: 5 -> 4 -> 3 -> 2 -> 1 -> G
    states  = [5, 4, 3, 2, 1, 'G']
    rewards = [0, -0.1, -0.1, 0, 1]        # reward 1 only on the last transition to G

    V_init = {1: 0.0, 2: 0.0, 3: -1.0, 4: 0.0, 5: 0.0, 'G': 0.0}
    gamma, lam, alpha = 0.9, 0.5, 0.1

    V_new = TD_lambda_tabular_update(states, rewards, V_init.copy(),
                                     gamma=gamma, lam=lam, alpha=alpha,
                                     terminal_value=0.0)

    print("Updated values (after TD(λ) on this episode):")
    for s in [5, 4, 3, 2, 1, 'G']:
        print(f"  V({s}) = {V_new[s]:.6f}")



    # # Episode path: 5 -> 4 -> 3 -> 2 -> 1 -> G
    # states  = [5, 4, 3, 2, 1, 'G']
    # rewards = [0, 0, 0, 0, 1]        # reward 1 only on the last transition to G
    #
    # V_init = {1: 10.0, 2: 15.0, 3: 20.0, 4: 25.0, 5: 30.0, 'G': 0.0}
    # gamma, lam, alpha = 0.9, 0.3, 0.1
    #
    # V_new = TD_lambda_tabular_update(states, rewards, V_init.copy(),
    #                                  gamma=gamma, lam=lam, alpha=alpha,
    #                                  terminal_value=0.0)
    #
    # print("Updated values (after TD(λ) on this episode):")
    # for s in [5, 4, 3, 2, 1, 'G']:
    #     print(f"  V({s}) = {V_new[s]:.6f}")


if __name__ == "__main__":
    main()

"""
Expected (rounded):
  V(5) ≈ 28.9970
"""

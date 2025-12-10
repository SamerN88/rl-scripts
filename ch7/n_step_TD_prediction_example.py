"""
Ch7 HW part 2: n-step TD prediction for a labeled gridworld trajectory.

We have a fixed trajectory of states (numbers are labels on the path):

    S_0.. = [5, 4, 3, 2, 1, G]    # G = terminal goal state

Rewards:
    r = 0 on all steps except the final transition into G, where r = 1.
    So the rewards for transitions along this trajectory are:
        R_1.. = [0, 0, 0, 0, 1]

Initial state values:
    V(1) = 10
    V(2) = 15
    V(3) = 20
    V(4) = 25
    V(5) = 30

Discount and step-size:
    gamma = 0.9
    alpha = 0.1

We perform ONE n-step TD update with n = 3 for the starting state S_0 = 5.

Algorithm used: n-step TD prediction (Sutton & Barto, Ch. 7)

n-step return:
    G_t^(n) = R_{t+1} + gamma*R_{t+2} + ... + gamma^{n-1}*R_{t+n}
              + gamma^n * V(S_{t+n})

n-step TD update:
    V(S_t) <- V(S_t) + alpha * ( G_t^(n) - V(S_t) )

For this question:
    t = 0, n = 3, S_0 = 5, S_3 = 2
    We must compute the new V(5).
"""


def n_step_return(states, rewards, V, t, n, gamma):
    """
    Compute the n-step return G_t^(n) for a single trajectory.

    states  : [S_0, S_1, ..., S_T]   (includes terminal at the end)
    rewards : [R_1, R_2, ..., R_T]   (reward for S_{k-1} -> S_k)
    V       : dict mapping nonterminal states -> value
    t       : starting time index (0-based, S_t is states[t])
    n       : number of steps for n-step TD
    gamma   : discount factor

    Uses:
        G_t^(n) = sum_{k=0}^{n-1} gamma^k * R_{t+1+k}
                  + gamma^n * V(S_{t+n})
    """
    G = 0.0
    power = 1.0  # gamma^0 initially

    # sum of discounted rewards
    for k in range(n):
        R_tp1_k = rewards[t + k]   # R_{t+1+k}
        G += power * R_tp1_k
        power *= gamma

    # bootstrap from state S_{t+n} (assumed nonterminal here)
    S_tn = states[t + n]
    G += power * V[S_tn]

    return G


def main():
    # Trajectory and rewards from the problem
    states = [5, 4, 3, 2, 1, "G"]   # S_0..S_5
    rewards = [0, 0, 0, 0, 1]       # R_1..R_5

    # Initial value function
    V = {
        1: 10.0,
        2: 15.0,
        3: 20.0,
        4: 25.0,
        5: 30.0,
    }

    gamma = 0.9
    alpha = 0.1
    n = 3
    t = 0                  # starting from state 5
    s_t = states[t]        # s_t = 5

    # ---- n-step TD update for state 5 ----
    G = n_step_return(states, rewards, V, t, n, gamma)
    old_value = V[s_t]
    new_value = old_value + alpha * (G - old_value)

    print(f"n-step return G_0^(3): {G:.5f}")
    print(f"Old V(5): {old_value:.5f}")
    print(f"New V(5): {new_value:.2f}")  # expected 28.09


if __name__ == "__main__":
    main()

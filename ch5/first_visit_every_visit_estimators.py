"""
Ch5 HW: single nonterminal state, single action.
One observed episode of length T with rewards r_t and gamma.
Compute:
  - first-visit MC estimator V(s)
  - every-visit MC estimator V(s)

For the quiz question:
  T = 10, r_t = +1 for all t, gamma = 1
  => first-visit V(s) = 10, every-visit V(s) = 5.5
"""


def compute_returns(rewards, gamma=1.0):
    """
    Given rewards [R_1, ..., R_T], compute returns [G_0, ..., G_{T-1}]
    for a single episode:
        G_t = R_{t+1} + gamma * G_{t+1}
    """
    T = len(rewards)
    G = [0.0] * T
    G_next = 0.0
    for t in range(T - 1, -1, -1):
        G_next = rewards[t] + gamma * G_next
        G[t] = G_next
    return G


def mc_estimators_single_state(rewards, gamma=1.0):
    """
    For this MDP there is only one nonterminal state s and it is visited
    at every time step until termination.

    First-visit MC estimator:
        V_first(s) = G_0

    Every-visit MC estimator:
        V_every(s) = average of G_t over all visits t = 0..T-1
    """
    G = compute_returns(rewards, gamma)
    V_first = G[0]
    V_every = sum(G) / len(G)
    return V_first, V_every


def main():
    # Quiz setup: 10-step episode, reward +1 each step, gamma = 1
    T = 10
    rewards = [1.0] * T
    gamma = 1.0

    V_first, V_every = mc_estimators_single_state(rewards, gamma)

    print(f"First-visit estimator V(s)  = {V_first}")
    print(f"Every-visit estimator V(s) = {V_every}")


if __name__ == "__main__":
    main()

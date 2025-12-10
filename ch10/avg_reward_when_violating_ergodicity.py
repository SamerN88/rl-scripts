"""
Ch10 HW, Part 2: Average reward for an alternating reward sequence.

Problem (paraphrased):
- Under any policy, the MDP produces the deterministic reward sequence
      +1, 0, +1, 0, +1, 0, ...
- This violates ergodicity, so the usual limit
      lim_{t->inf} E[R_t | S_0, A_0:t-1 ~ pi]
  does not exist.
- Nevertheless, the *average reward* is well-defined.

Average-reward criterion (Eq. 10.6 in Sutton & Barto):
    r_bar(pi) = lim_{h->inf} (1/h) * sum_{t=1}^h E[ R_t | S_0, A_0:t-1 ~ pi ]

For a deterministic periodic sequence with period k and rewards
    (r_1, r_2, ..., r_k) repeating forever, the long-run average reward is
    r_bar = (1/k) * sum_{i=1}^k r_i

Here:
    period = 2, rewards = (1, 0)
    => r_bar = (1/2) * (1 + 0) = 0.5
"""


def average_reward_periodic(rewards_per_cycle):
    """
    Compute long-run average reward for a deterministic periodic sequence.

    rewards_per_cycle : list of rewards in one full cycle, e.g. [1, 0]
    Returns:
        long-run average reward r_bar
    """
    total = sum(rewards_per_cycle)
    k = len(rewards_per_cycle)
    return total / k


def main():
    # Alternating +1, 0, +1, 0, ... corresponds to the cycle [1, 0]
    cycle = [1.0, 0.0]
    r_bar = average_reward_periodic(cycle)

    # Print to two decimal places as required by the question
    print(f"Average reward r_bar = {r_bar:.2f}")   # expected: 0.50


if __name__ == "__main__":
    main()

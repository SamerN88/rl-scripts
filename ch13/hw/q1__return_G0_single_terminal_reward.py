"""
Ch. 13 HW Part 2, Q1 – Discounted return at start state for REINFORCE trajectory.

We have a single episode (trajectory) for an agent using REINFORCE in a
continuous 2D maze. Rewards are:
    R(s_0, a_0) = 0
    R(s_1, a_1) = 0
    R(s_2, a_2) = 0
    R(s_3, a_3) = 0
    R(s_4, a_4) = -1        (terminal trap at s_5)
Discount factor:
    gamma = 0.9

We want the return G at state s_0 (the start state).

Algorithm: Monte Carlo return (discounted return)
-----------------------------------------------
For a finite episode with terminal time T and rewards R_{t+1}:
    G_t = sum_{k=t}^{T-1} (gamma^(k - t)) * R_{k+1}

In this question:
    T = 5, and only the last reward is non-zero:
    R_{5} = -1, all others = 0.

Therefore:
    G_0 = gamma^4 * (-1)

This script computes G_0 numerically and prints it to 4 decimal places.
"""

def main():
    gamma = 0.9

    # rewards R_{t+1} for t = 0..4 (from s_t, a_t to s_{t+1})
    rewards = [0.0, 0.0, 0.0, 0.0, -1.0]

    G0 = 0.0
    for t, r in enumerate(rewards):
        G0 += (gamma ** t) * r

    print(f"G at state s0: {G0:.4f}")


if __name__ == "__main__":
    main()

def compute_returns(gamma, rewards):
    """
    Compute G_0, G_1, ..., G_T for a single trajectory.

    Args:
        gamma   : discount factor (float)
        rewards : list of rewards [R_1, R_2, ..., R_T]

    Returns:
        list of returns [G_0, G_1, ..., G_T]
        (G_T is 0 because there are no rewards after time T)
    """
    T = len(rewards)
    G = [0.0] * (T + 1)          # G_T = 0 by definition

    # Backward recursion: G_t = R_{t+1} + gamma * G_{t+1}
    for t in range(T - 1, -1, -1):
        G[t] = rewards[t] + gamma * G[t + 1]

    return G


def main():
    # Problem setup: gamma = 0.5, rewards R1..R5, then 0 afterwards
    gamma = 0.5
    rewards = [-1, 2, 6, 3, 2]   # [R1, R2, R3, R4, R5]

    G = compute_returns(gamma, rewards)

    # Print G_0, G_1, ..., G_5
    for t, g in enumerate(G):
        print(f"G_{t} = {g}")


if __name__ == "__main__":
    main()

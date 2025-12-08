def nstep_return(episode, V, t, n, *, gamma):
    """
    Standard n-step return G_{t:t+n} for episodic MRP with bootstrapping.
    episode: list[(S_t, R_{t+1})] ending at last (S_{T-1}, R_T).
    """
    T = len(episode)
    G = 0.0
    disc = 1.0
    for k in range(n):
        if t + k >= T:
            break
        _, R_next = episode[t + k]
        G += disc * R_next
        disc *= gamma
    # Bootstrap if we still have a following state inside episode
    if t + n < T:
        S_next, _ = episode[t + n]
        G += disc * V[S_next]
    return G


def main():
    # Example episode and values (same structure as your exam problems)
    episode = [(0, 1), (1, 0), (2, 1), (3, 0), (4, 1)]  # [(S_t, R_{t+1})...]
    V = {0: 0.0, 1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}
    gamma = 0.9

    t = 0  # compute returns starting from state/time index 0
    for n in [1, 2, 3, 4, 5]:
        g = nstep_return(episode, V, t, n, gamma=gamma)
        print(f"G_{{{t}:{t}+{n}}} (n={n}) = {g:.6f}")


if __name__ == "__main__":
    main()

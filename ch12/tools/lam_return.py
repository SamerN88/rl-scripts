def compute_lam_return(s, episode, V, *, gamma, lam):
    """
    Compute the forward-view lambda return for the given state s.
    episode: list of (S_t, R_{t+1}) pairs up to terminal (R after each S)
    V: dict of state-value estimates
    """
    # find the time index where S_t == s
    t = next(i for i, (S, _) in enumerate(episode) if S == s)
    T = len(episode)
    G_tn = []

    # compute n-step returns from time t
    for n in range(1, T - t + 1):
        G = 0.0
        discount = 1.0
        for k in range(n):
            _, R_next = episode[t + k]
            G += discount * R_next
            discount *= gamma
        # bootstrap if not at terminal
        if t + n < T:
            S_next, _ = episode[t + n]
            G += discount * V[S_next]
        G_tn.append(G)

    # conventional return (full)
    G_t = sum((gamma ** i) * episode[t + i][1] for i in range(T - t))

    # compute lambda-return
    G_lam = (1 - lam) * sum((lam ** (n - 1)) * G_tn[n - 1] for n in range(1, len(G_tn) + 1)) + (lam ** (T - t)) * G_t

    return G_lam


def main():
    # Example data
    #         [(s0, r1), (s1, r2), (s2, r3), (s3, r4), (s4, r5)]
    episode = [(0,  1),  (1,  0),  (2,  1),  (3,  0),  (4,  1)]
    V = {0: 0, 1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}

    s = 0
    lam_return = compute_lam_return(s, episode, V, gamma=0.9, lam=0.5)
    print(f'lambda-return for state {s}: {lam_return}')


if __name__ == '__main__':
    main()

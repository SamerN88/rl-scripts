from nstep_return import nstep_return


def compute_nstep_truncated_TD_lam_return(s, episode, V, h, *, gamma, lam):
    """
    Compute the truncated n-step TD(λ) target for state s with horizon h.
    Definition (episodic, per Sutton & Barto Ch.12):
        G_t^{λ|h} = (1-λ) * sum_{n=1}^{h-1} λ^{n-1} G_{t:t+n}  +  λ^{h-1} G_{t:t+h}
    If the episode ends before t+h, G_{t:t+h} collapses to the conventional return.
    """
    # locate time index t of state s
    t = next(i for i, (S, _) in enumerate(episode) if S == s)
    # accumulate first h-1 weighted n-step returns
    total = 0.0
    for n in range(1, h):
        total += (lam ** (n - 1)) * nstep_return(episode, V, t, n, gamma=gamma)
    total *= (1 - lam)
    # tail uses the h-step return (single term), weighted by λ^{h-1}
    tail = (lam ** (h - 1)) * nstep_return(episode, V, t, h, gamma=gamma)
    return total + tail


def main():
    # Example episode and values (same structure as your exam problems)
    episode = [(0, 1), (1, 0), (2, 1), (3, 0), (4, 1)]  # [(S_t, R_{t+1})...]
    V = {0: 0.0, 1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}
    gamma = 0.9
    lam = 0.5
    h = 3
    s = 0

    # find time index t of s
    t = next(i for i, (S, _) in enumerate(episode) if S == s)

    print("=== Truncated n-step TD(λ) Example ===")
    print(f"state s = {s}, t = {t}, horizon h = {h}, gamma = {gamma}, lambda = {lam}")
    print()

    # 1) Show each G_{t:t+n} for n=1..h
    Gs = {}
    for n in range(1, h + 1):
        Gs[n] = nstep_return(episode, V, t, n, gamma=gamma)
        print(f"G_{{{t}:{t}+{n}}} (n={n}) = {Gs[n]:.6f}")

    print()

    # 2) Show weights and contributions for n=1..h-1 with (1-λ)λ^{n-1}
    print("Contributions (prefix terms):")
    prefix_sum = 0.0
    for n in range(1, h):
        w = (1 - lam) * (lam ** (n - 1))
        contrib = w * Gs[n]
        prefix_sum += contrib
        print(
            f"  n={n}: weight=(1-λ)λ^{n-1}={(1-lam):.6f}*{(lam**(n-1)):.6f}={w:.6f}, "
            f"G={Gs[n]:.6f}, contrib={contrib:.6f}"
        )

    print(f"Prefix total = {prefix_sum:.6f}")
    print()

    # 3) Tail term: λ^{h-1} * G_{t:t+h}
    tail_w = lam ** (h - 1)
    tail_contrib = tail_w * Gs[h]
    print(
        f"Tail term: λ^{h-1}={tail_w:.6f} * G_{{{t}:{t}+{h}}}={Gs[h]:.6f} "
        f"=> contrib={tail_contrib:.6f}"
    )

    # 4) Final truncated TD(λ) return
    G_trunc = prefix_sum + tail_contrib
    print()
    print(f"Truncated TD(λ) return G_t^(λ|h) = {G_trunc:.6f}")

    # Also show the direct function call result for sanity
    direct = compute_nstep_truncated_TD_lam_return(s, episode, V, h, gamma=gamma, lam=lam)
    print(f"(Direct function result)        = {direct:.6f}")


if __name__ == '__main__':
    main()

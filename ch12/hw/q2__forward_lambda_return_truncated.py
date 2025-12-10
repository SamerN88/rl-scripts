"""
Ch12 HW, Part 2, Q2: n-step *truncated* TD(lambda) return for state 0.

Same episode as Q1:

    S0 = 0, R1 = 1,
    S1 = 1, R2 = 0,
    S2 = 2, R3 = 1,
    S3 = 3, R4 = 0,
    S4 = 4, R5 = 1, then terminal.

State values:
    V(0) = 0.0
    V(1) = 0.1
    V(2) = 0.2
    V(3) = 0.3
    V(4) = 0.4

Parameters:
    gamma  = 0.9
    lambda = 0.5
    horizon h = 3
    We want the truncated TD(lambda) return for state 0 (t = 0).

------------------------------------------------------------
Algorithm / formula used: n-step *truncated* lambda-return.

The n-step truncated lambda-return at horizon h (for state S_t) is:

    G_{t:h}^lambda = (1 - lambda) * sum_{n=1}^{h - t - 1} lambda^{n-1} * G_{t:t+n}
                     + lambda^{h - t - 1} * G_{t:h}

Here:
    - G_{t:t+n} is the n-step return bootstrapping from V(S_{t+n})
    - h is the horizon index (in time), so for t = 0 and h = 3
      we use 1-step, 2-step, and 3-step returns.

n-step return (for n >= 1):

    G_{t:t+n} = R_{t+1} + gamma * R_{t+2} + ... + gamma^{n-1} * R_{t+n}
                + gamma^n * V(S_{t+n})

For this specific question (t = 0, h = 3):

1-step, 2-step, 3-step returns (re-using Q1’s results):

    G_{t+1}   = G_{0:1} = R1 + gamma * V(S1)
              = 1 + 0.9 * 0.1
              = 1.09

    G_{t:t+2} = G_{0:2} = R1 + gamma * R2 + gamma^2 * V(S2)
              = 1 + 0.9 * 0 + 0.9^2 * 0.2
              = 1.162

    G_{t:t+3} = G_{0:3} = R1 + gamma * R2 + gamma^2 * R3
                           + gamma^3 * V(S3)
              = 1 + 0.9 * 0 + 0.9^2 * 1 + 0.9^3 * 0.3
              = 2.0287   (approximately)

Using the truncated lambda-return definition with lambda = 0.5 and h = 3:

    G_0^lambda (truncated) =
        (1 - lambda) * [ G_{t+1} + lambda * G_{t:t+2} ]
        + lambda^2 * G_{t:t+3}

    = (1 - 0.5) * [ 1.09 + 0.5 * 1.162 ] + 0.5^2 * 2.0287
    = 0.5 * [ 1.09 + 0.581 ] + 0.25 * 2.0287
    = 0.5 * 1.671 + 0.507175
    = 0.8355 + 0.507175
    = 1.342675  (approximately)

The code below computes this programmatically.
"""


def n_step_return(states, rewards, V, gamma, t, n):
    """
    Compute G_{t:t+n}, the n-step return bootstrapping from V(S_{t+n}).

    states  : [S_0, S_1, ..., S_T]   (includes terminal marker at the end)
    rewards : [R_1, R_2, ..., R_T]
    V       : dict mapping nonterminal state -> value
    gamma   : discount factor
    t       : starting time index (0-based, for S_t)
    n       : number of steps for the n-step return
    """
    G = 0.0
    power = 1.0  # gamma^0

    # discounted rewards up to n steps
    for k in range(n):
        G += power * rewards[t + k]  # R_{t+1+k}
        power *= gamma

    # bootstrap from V(S_{t+n})
    s_tn = states[t + n]
    G += power * V[s_tn]
    return G


def truncated_lambda_return(states, rewards, V, gamma, lam, t, h):
    """
    Compute the n-step *truncated* lambda-return G_{t:h}^lambda.

    Using:
        G_{t:h}^lambda =
            (1 - lam) * sum_{n=1}^{h - t - 1} lam^{n-1} * G_{t:t+n}
            + lam^{h - t - 1} * G_{t:h}

    Here h is treated as "t + horizon_steps".
    For this problem we call with t = 0, h = 3 (3-step horizon).
    """
    # Number of steps from t to horizon
    H = h  # interpret h as "h steps ahead from t"

    # First term: (1 - lambda) * sum_{n=1}^{H-1} lambda^{n-1} G_{t:t+n}
    G_lambda = 0.0
    lam_power = 1.0  # lam^{n-1}, start at lam^0 for n=1
    for n in range(1, H):
        G_n = n_step_return(states, rewards, V, gamma, t, n)
        G_lambda += lam_power * G_n
        lam_power *= lam

    G_lambda *= (1.0 - lam)

    # Final term: lambda^{H-1} * G_{t:H}
    G_H = n_step_return(states, rewards, V, gamma, t, H)
    G_lambda += (lam ** (H - 1)) * G_H

    return G_lambda


def main():
    # Episode and values (same as Q1)
    states = [0, 1, 2, 3, 4, "terminal"]
    rewards = [1, 0, 1, 0, 1]

    V = {0: 0.0, 1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}

    gamma = 0.9
    lam = 0.5
    t = 0
    horizon_steps = 3  # h = 3

    G_trunc = truncated_lambda_return(states, rewards, V, gamma, lam, t, horizon_steps)
    print(f"G_0^lambda (truncated, h=3) = {G_trunc:.6f}")  # expected 1.342675


if __name__ == "__main__":
    main()

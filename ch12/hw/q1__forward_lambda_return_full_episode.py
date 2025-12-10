"""
Ch12 HW, Part 2, Q1: Forward-view lambda-return for state 0 in an MRP.

Episode:
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

Discount and trace-decay:
    gamma = 0.9
    lambda = 0.5

Goal:
    Compute the forward-view lambda-return G_0^lambda for state 0,
    using the definition in Sutton & Barto, Eq. (12.3).

------------------------------------------------------------
Definition (forward-view lambda-return):

For an episode with T time-steps (here T = 5 rewards), the lambda-return
for state S_t is:

    G_t^lambda = (1 - lambda) * sum_{n=1}^{T - t - 1} lambda^{n-1} * G_t^{(n)}
                 + lambda^{T - t - 1} * G_t

where:
    - G_t^{(n)} is the n-step return bootstrapping from V at S_{t+n}
    - G_t is the full Monte Carlo (no bootstrapping) return

n-step return (for n < T - t):

    G_t^{(n)} = R_{t+1} + gamma * R_{t+2} + ... + gamma^{n-1} * R_{t+n}
                + gamma^n * V(S_{t+n})

Full return:

    G_t = R_{t+1} + gamma * R_{t+2} + ... + gamma^{T - t - 1} * R_T

------------------------------------------------------------
For this specific question (t = 0):

1-step, 2-step, 3-step, 4-step returns:

    G_{t+1} = R1 + gamma * V(S1)
             = 1 + 0.9 * 0.1
             = 1.09

    G_{t+2} = R1 + gamma * R2 + gamma^2 * V(S2)
             = 1 + 0.9 * 0 + 0.9^2 * 0.2
             = 1.162

    G_{t+3} = R1 + gamma * R2 + gamma^2 * R3 + gamma^3 * V(S3)
             = 1 + 0.9 * 0 + 0.9^2 * 1 + 0.9^3 * 0.3
             = 2.0287  (approximately)

    G_{t+4} = R1 + gamma * R2 + gamma^2 * R3 + gamma^3 * R4
              + gamma^4 * V(S4)
             = 1 + 0.9 * 0 + 0.9^2 * 1 + 0.9^3 * 0 + 0.9^4 * 0.4
             = 2.07244 (approximately)

Full discounted return:

    G_t = R1 + gamma * R2 + gamma^2 * R3 + gamma^3 * R4 + gamma^4 * R5
         = 1 + 0.9 * 0 + 0.9^2 * 1 + 0.9^3 * 0 + 0.9^4 * 1
         = 2.4661 (approximately)

Putting everything together (Eq. 12.3) with lambda = 0.5:

    G_0^lambda = (1 - 0.5) * (
                    G_{t+1}
                    + 0.5 * G_{t+2}
                    + 0.5^2 * G_{t+3}
                    + 0.5^3 * G_{t+4}
                 )
                 + 0.5^4 * G_t

               = 1.37274625 (approximately)

The script below computes this value programmatically.
"""


def n_step_return(states, rewards, V, gamma, t, n):
    """
    Compute G_t^{(n)} for a single episode, bootstrapping from V(S_{t+n}).

    states  : [S_0, S_1, ..., S_T]  (S_T is terminal marker or last state)
    rewards : [R_1, R_2, ..., R_T]
    V       : dict mapping nonterminal state -> V(state)
    gamma   : discount factor
    t       : time index of starting state S_t (0-based)
    n       : number of steps for the n-step return

    Assumes t + n <= T and S_{t+n} is nonterminal for n < T - t.
    """
    G = 0.0
    power = 1.0  # gamma^0
    # discounted rewards up to n steps
    for k in range(n):
        G += power * rewards[t + k]   # R_{t+1+k}
        power *= gamma
    # bootstrap from V at S_{t+n}
    s_tn = states[t + n]
    G += power * V[s_tn]
    return G


def full_return(states, rewards, gamma, t):
    """
    Compute the full Monte Carlo return G_t (no bootstrapping).
    """
    G = 0.0
    power = 1.0
    T = len(rewards)
    for k in range(T - t):
        G += power * rewards[t + k]   # R_{t+1+k}
        power *= gamma
    return G


def lambda_return_forward(states, rewards, V, gamma, lam, t):
    """
    Compute G_t^lambda using the forward-view definition:

    G_t^lambda = (1 - lam) * sum_{n=1}^{T - t - 1} lam^{n-1} * G_t^{(n)}
                 + lam^{T - t - 1} * G_t
    """
    T = len(rewards)  # number of rewards
    # number of n-step terms before the full return
    max_n = T - t - 1

    # sum over n-step returns
    G_lambda = 0.0
    lam_power = 1.0  # lam^{n-1}, starts at lam^0 for n=1
    for n in range(1, max_n + 1):
        G_n = n_step_return(states, rewards, V, gamma, t, n)
        G_lambda += lam_power * G_n
        lam_power *= lam

    G_lambda *= (1.0 - lam)

    # add final term with full return
    G_t_full = full_return(states, rewards, gamma, t)
    G_lambda += (lam ** max_n) * G_t_full

    return G_lambda


def main():
    # Episode specification
    states = [0, 1, 2, 3, 4, "terminal"]
    rewards = [1, 0, 1, 0, 1]

    # Value function
    V = {0: 0.0, 1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}

    gamma = 0.9
    lam = 0.5
    t = 0  # we want G_0^lambda

    G_lambda_0 = lambda_return_forward(states, rewards, V, gamma, lam, t)
    print(f"G_0^lambda = {G_lambda_0:.6f}")  # expected ~1.372746


if __name__ == "__main__":
    main()

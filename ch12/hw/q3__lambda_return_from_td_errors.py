"""
Ch12 HW, Part 2, Q3: Forward-view lambda-return from TD errors.

We use the TD-error formulation of the lambda-return:

    G_t^lambda  ≈  v_hat(S_t, w)  +  sum_{k=t}^{T-1} (gamma * lambda)^{k - t} * delta_k

where
    - delta_k is the TD error at time k
    - gamma is the discount factor
    - lambda is the trace decay parameter
    - v_hat(S_t, w) is the current value estimate for state S_t

In this question:
    TD errors from time t:   delta_t..delta_{t+4} = [1.1, 0.4, 0.8, -0.5, 0.7]
    Episode terminates after these 5 steps.
    v_hat(S_t, w) = 0
    gamma  = 0.9
    lambda = 0.5

So:

    G_t^lambda = 0
                 + (gamma*lambda)^0 * 1.1
                 + (gamma*lambda)^1 * 0.4
                 + (gamma*lambda)^2 * 0.8
                 + (gamma*lambda)^3 * (-0.5)
                 + (gamma*lambda)^4 * 0.7

with gamma*lambda = 0.9 * 0.5 = 0.45.

The expected numerical answer (to 6 decimals) is:

    G_t^lambda = 1.425142
"""


def lambda_return_from_td_errors(deltas, gamma, lam, v_hat_t=0.0):
    """
    Compute G_t^lambda from a sequence of TD errors.

    Args:
        deltas  : list of TD errors [delta_t, delta_{t+1}, ..., delta_{T-1}]
        gamma   : discount factor
        lam     : trace-decay parameter (lambda)
        v_hat_t : current value estimate v_hat(S_t, w)

    Returns:
        G_t^lambda (float)
    """
    g_lam = v_hat_t
    factor = 1.0          # (gamma*lambda)^{k-t} starts at power 0
    gl = gamma * lam

    for delta_k in deltas:
        g_lam += factor * delta_k
        factor *= gl

    return g_lam


def main():
    deltas = [1.1, 0.4, 0.8, -0.5, 0.7]
    gamma = 0.9
    lam = 0.5
    v_hat_t = 0.0

    g_lambda = lambda_return_from_td_errors(deltas, gamma, lam, v_hat_t)
    print(f"G_t^lambda = {g_lambda:.6f}")  # expected: 1.425142


if __name__ == "__main__":
    main()

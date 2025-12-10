"""
Ch. 13 HW Part 2, Q2 – REINFORCE update of W at first state s0; sum of elements.

Setting
-------
We continue the maze example from Q1:
- Policy: Gaussian with mean mu = W s  (linear function approximator)
- W is a 2x2 matrix, initialized to all zeros.
- State s0 = (0, 0).
- First action a0 = (0.5, -0.2) (sampled from N(mu, I)).
- Only the last reward is non-zero: R(s4, a4) = -1; all earlier rewards are 0.
- Discount factor: gamma = 0.9.
- Learning rate for REINFORCE: alpha = 0.1.

We are asked:
    After we loop over the first state s0 only (i.e., perform a single
    REINFORCE update using G0 and (s0, a0)), what is the *sum of all elements*
    of the updated W?  (Parameters have NOT yet been updated using s1, s2, ...)

Algorithms and formulas
-----------------------
1) Monte Carlo return used by REINFORCE:
   For terminal time T and rewards R_{t+1}:
       G_t = sum_{k=t}^{T-1} (gamma^(k - t)) * R_{k+1}

   In this episode (same as Q1):
       rewards = [0, 0, 0, 0, -1]
       => G_0 = gamma^4 * (-1)

2) REINFORCE policy-gradient update (episodic):
       W <- W + alpha * G_t * grad_W log pi(a_t | s_t)

3) Gaussian policy with identity covariance, mean mu = W s:
   PDF:   pi(a | s) = (1 / sqrt((2*pi)^d * |I|)) *
                      exp( -0.5 * (a - mu)^T I^{-1} (a - mu) )

   Log-pdf (ignoring constants):
       log pi(a | s) = -0.5 * (a - mu)^T (a - mu)

   Gradient wrt mu:
       grad_mu log pi(a | s) = a - mu

   Chain rule for mu = W s:
       grad_W log pi(a | s)
         = grad_mu log pi(a | s) * (d mu / d W)
         = (a - mu) s^T
         = (a - W s) s^T

4) Specialization to the first state s0:
   - For s0 = (0, 0), we have
         mu0 = W s0 = 0
     (because W is initially all zeros and s0 is the zero vector).

   - Therefore the gradient is
         grad_W log pi(a0 | s0)
           = (a0 - W s0) s0^T
           = (a0 - 0) * [0 0]      (outer product)
           = 0_{2x2}               (all zeros matrix)

   - REINFORCE update at s0:
         W_new = W + alpha * G0 * grad_W log pi(a0 | s0)
               = W + alpha * G0 * 0_{2x2}
               = W  (no change at all)

   Since W started as the zero matrix, W_new is also all zeros,
   and the sum of all elements in W_new is 0.

This script explicitly computes G0, grad_W log pi at s0, performs the
REINFORCE update, and prints the sum of all elements of W_new to 4 decimals.
"""

import numpy as np


def main():
    gamma = 0.9
    alpha = 0.1

    # Rewards R_{t+1} for t = 0..4
    rewards = [0.0, 0.0, 0.0, 0.0, -1.0]

    # Compute G_0
    G0 = 0.0
    for t, r in enumerate(rewards):
        G0 += (gamma ** t) * r

    # Initial parameters and first (state, action)
    W = np.zeros((2, 2))
    s0 = np.array([0.0, 0.0])
    a0 = np.array([0.5, -0.2])

    # Mean mu0 = W s0
    mu0 = W @ s0

    # grad_W log pi(a0 | s0) = (a0 - mu0) s0^T
    grad_log_pi_W = np.outer(a0 - mu0, s0)

    # REINFORCE update using only (s0, a0, G0)
    W_new = W + alpha * G0 * grad_log_pi_W

    # Sum of all elements in updated W
    sum_W_elements = float(W_new.sum())
    print(f'W = \n{W_new}\n')
    print(f"Sum of all elements in updated W after s0 update: {sum_W_elements:.4f}")


if __name__ == "__main__":
    main()

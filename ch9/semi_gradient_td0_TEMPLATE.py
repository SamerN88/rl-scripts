"""
Semi-gradient TD(0) – General Linear Function Approximation Helper Script
=======================================================================

Purpose
-------
This script implements the Semi-gradient TD(0) algorithm from Sutton & Barto
(2nd ed.), Section 9.4 / Algorithm "Semi-gradient TD(0) for estimating v_hat ≈ v_pi".

It is written as an EXAM HELPER:
- You only need to fill in the USER-EDIT SECTION with:
    * discount factor gamma
    * step size alpha
    * initial weight vector w
    * feature vectors x(s) for each state
    * list of transitions (S, R, S', terminal_next)
- Then run the script to get the updated weights after one pass over the data.

Algorithm (linear value function)
---------------------------------
We assume a linear value function:
    v_hat(s, w) = w^T x(s)

For each time step t with transition (S_t, R_{t+1}, S_{t+1}):
    v_t     = v_hat(S_t, w) = w^T x(S_t)
    v_tp1   = v_hat(S_{t+1}, w) if S_{t+1} is non-terminal, else 0
    delta_t = R_{t+1} + gamma * v_tp1 - v_t

Gradient for linear v_hat:
    grad_w v_hat(S_t, w) = x(S_t)

Semi-gradient TD(0) update:
    w <- w + alpha * delta_t * x(S_t)

This script:
------------
- Applies the above update once for EACH transition in TRANSITIONS, IN ORDER.
- Prints:
    * per-step TD error delta_t
    * the new weights after each step
    * the final weight vector at the end

How to use during an exam
-------------------------
1) Replace the USER-EDIT SECTION with the example's numbers:
   - Put the given gamma, alpha, and initial w.
   - Define FEATURES[state] = x(s) for each state.
   - List transitions as (state, reward, next_state, is_terminal_next).

2) Run:
       python semi_gradient_td0_helper.py

3) Read off the final weights w (often what the question asks for).

NOTE: States can be integers or strings; just be consistent between FEATURES
      and TRANSITIONS.
"""

import numpy as np

# ========================= USER-EDIT SECTION ========================= #

# Discount factor and step size from the problem
GAMMA = 0.95
ALPHA = 0.1

# Initial weight vector w (1D numpy array)
# Example placeholder for d = 3 features:
#   w = [w0, w1, w2]
w = np.array([0.0, 0.0, 0.0])  # <-- EDIT THIS LINE for the exam problem

# Feature vectors x(s) for each state s.
# Replace the keys and vectors with those given in the exam.
# Example with states 'A', 'B', 'C' and 3 features:
FEATURES = {
    'A': np.array([1.0, 0.0, 0.0]),
    'B': np.array([0.0, 1.0, 0.0]),
    'C': np.array([0.0, 0.0, 1.0]),
    # Add/modify states and feature vectors as needed.
}

# List of transitions for ONE PASS.
# Each item is a tuple: (state, reward, next_state, is_terminal_next)
#
# Example placeholder transitions:
#   A --(R=1)--> B
#   B --(R=2)--> C
#   C --(R=0)--> terminal
TRANSITIONS = [
    ('A', 1.0, 'B', False),
    ('B', 2.0, 'C', False),
    ('C', 0.0, None, True),
]

# ======================= END USER-EDIT SECTION ======================= #


def phi(state):
    """
    Return feature vector x(s) for state s.

    For terminal states, we never call this: we use v_hat(terminal) = 0.
    """
    return FEATURES[state]


def v_hat(state, weights):
    """Linear value function: v_hat(s, w) = w^T x(s)."""
    return float(weights @ phi(state))


def semi_gradient_td0_pass(weights, transitions, gamma, alpha):
    """
    Perform ONE pass of semi-gradient TD(0) over a list of transitions.

    transitions: list of (S, R, S_next, is_terminal_next)
    returns: updated weights
    """
    w = weights.copy()

    print("Initial weights:", w)

    for step, (s, r, s_next, terminal_next) in enumerate(transitions, start=1):
        v_s = v_hat(s, w)

        if terminal_next:
            v_next = 0.0
        else:
            v_next = v_hat(s_next, w)

        delta = r + gamma * v_next - v_s
        x_s = phi(s)

        w += alpha * delta * x_s

        print(f"\nStep {step}:")
        print(f"  S = {s}, R = {r}, S' = {s_next}, terminal_next = {terminal_next}")
        print(f"  v(S)   = {v_s:.6f}")
        print(f"  v(S')  = {v_next:.6f}")
        print(f"  delta  = {delta:.6f}")
        print(f"  new w  = {w}")

    return w


def main():
    final_w = semi_gradient_td0_pass(
        weights=w,
        transitions=TRANSITIONS,
        gamma=GAMMA,
        alpha=ALPHA,
    )

    print("\nFinal weights after one TD(0) pass:")
    for i, wi in enumerate(final_w):
        print(f"w[{i}] = {wi:.6f}")


if __name__ == "__main__":
    main()

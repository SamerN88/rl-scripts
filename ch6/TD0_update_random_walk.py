"""
Ch6 HW, Part 2: Random-walk Markov Reward Process (A–E between terminals).

Question (paraphrased):
- Initial values: V(A)=V(B)=...=V(E)=0.5, V(terminal)=0.
- Learning rate alpha = 0.1, gamma = 1.
- In the first episode, state A transitions directly to the terminal
  with reward R = 0.
- Using TD(0), what is the CHANGE in the value of A (signed)?

TD(0) update:
    V(s) <- V(s) + alpha * ( R + gamma * V(s') - V(s) )
Change:
    delta = alpha * ( R + gamma * V(s') - V(s) )
"""


def td0_update(v_s, r, v_s_next, alpha, gamma=1.0):
    """One-step TD(0) update for a single state s."""
    target = r + gamma * v_s_next
    delta = alpha * (target - v_s)
    v_new = v_s + delta
    return v_new, delta


def main():
    # Problem setup for state A in this first episode
    v_A_initial = 0.5   # initial V(A)
    v_terminal = 0.0    # V(terminal)
    r = 0.0             # reward on transition A -> terminal
    alpha = 0.1
    gamma = 1.0

    v_A_new, delta_A = td0_update(v_A_initial, r, v_terminal, alpha, gamma)

    print(f"Change in V(A): {delta_A}")      # should be -0.05
    print(f"New value V(A): {v_A_new}")            # should be 0.45


if __name__ == "__main__":
    main()

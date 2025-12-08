# chapter4_gridworld_q.py

"""
This code solves Q1 from Ch4 HW Part 1.

Q2 says: Suppose a new state 15 is added below state 13. Its actions (left, up, right, down) go to states
12, 13, 14, and 15, respectively. Transitions from all original states are unchanged.
What is v_pi(15) under the equiprobable random policy?

Answer: v_pi(15) = -20

Explanation (Bellman expectation equation):
For an equiprobable random policy in this undiscounted task (gamma = 1), reward r = -1 on every transition and
each action from 15 is chosen with probability 1/4. The Bellman equation for state values is

    v_pi(s) = sum_a pi(a|s) * [ r + gamma * v_pi(s') ]

For state 15:
    v_pi(15) = -1 + (1/4) * ( v_pi(12) + v_pi(13) + v_pi(14) + v_pi(15) )

Using the known values v_pi(12) = -22, v_pi(13) = -20, v_pi(14) = -14:

    v_pi(15) = -1 + (1/4) * ( -22 - 20 - 14 + v_pi(15) )

Solve for v_pi(15) to get:
    v_pi(15) = -20
"""

# State values v_pi from Example 4.1 (row-major, states 0..15)
STATE_VALUES = [
    0.0, -14.0, -20.0, -22.0,
    -14.0, -18.0, -20.0, -20.0,
    -20.0, -20.0, -18.0, -14.0,
    -22.0, -20.0, -14.0,  0.0,
]

GRID_ROWS = 4
GRID_COLS = 4
TERMINAL_STATES = {0, 15}      # top-left and bottom-right in the 4x4
REWARD_PER_STEP = -1.0
GAMMA = 1.0                    # undiscounted task

ACTIONS = {
    "up":    (-1,  0),
    "down":  ( 1,  0),
    "left":  ( 0, -1),
    "right": ( 0,  1),
}


def next_state(s, action):
    """
    Deterministic transition for the 4x4 gridworld.
    If the move would go off-grid, the state stays the same.
    """
    if s in TERMINAL_STATES:
        return s

    row, col = divmod(s, GRID_COLS)
    dr, dc = ACTIONS[action]
    nr, nc = row + dr, col + dc

    # If off-grid, stay in the same state
    if nr < 0 or nr >= GRID_ROWS or nc < 0 or nc >= GRID_COLS:
        return s

    return nr * GRID_COLS + nc


def q_value(s, action, v=STATE_VALUES, gamma=GAMMA, r_step=REWARD_PER_STEP):
    """
    Compute q_pi(s, a) = E[ R_{t+1} + gamma * v_pi(S_{t+1}) | S_t=s, A_t=a ]
    for this deterministic gridworld.
    """
    s_next = next_state(s, action)
    return r_step + gamma * v[s_next]


def main():
    # Question: under the equiprobable random policy, what is q(7, down)?
    s = 7
    action = "down"
    q_7_down = q_value(s, action)
    print(f"q(7, down) = {q_7_down}")


if __name__ == "__main__":
    main()

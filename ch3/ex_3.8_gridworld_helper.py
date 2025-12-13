"""
Ch. 3 (Example 3.8 / Eq. 3.8 style) – Return for a periodic reward pattern
========================================================================

Problem form (like the Gridworld A example)
-------------------------------------------
You start in some state and then the rewards you receive follow a *repeating*
pattern, often something like:

- You get reward = R every P time steps (e.g., every 5 steps),
- and all other rewards are 0.

You want the return:
    G_0 = sum_{k=0}^∞ gamma^k * R_{k+1}

Key math (geometric series)
---------------------------
If the first nonzero reward happens after `offset` steps (offset >= 0),
and then repeats every `period` steps:

    R_{offset+1} = R
    R_{offset+period+1} = R
    R_{offset+2*period+1} = R
    ...

Then:
    G_0 = R * gamma^offset * (1 + gamma^period + gamma^(2*period) + ...)
        = R * gamma^offset * [ 1 / (1 - gamma^period) ]

Special case in the screenshot:
- gamma = 0.9
- reward R = 10
- period P = 5
- offset = 0   (because R1=10 is immediate from state A)
So:
    G0 = 10 / (1 - 0.9^5) = 24.419...

This script lets you plug in (gamma, reward, period, offset) and prints G0.

Usage
-----
Edit the USER-EDIT section, then run:
    python periodic_return_helper.py
"""

def periodic_return(gamma: float, reward: float, period: int, offset: int = 0) -> float:
    """
    Closed-form periodic-return:
        G0 = reward * gamma^offset / (1 - gamma^period)
    """
    if period <= 0:
        raise ValueError("period must be a positive integer")
    if offset < 0:
        raise ValueError("offset must be >= 0")
    if not (0.0 <= gamma < 1.0):
        raise ValueError("gamma must be in [0, 1) for the infinite sum to converge")

    denom = 1.0 - (gamma ** period)
    return reward * (gamma ** offset) / denom


def main():
    # ========================= USER-EDIT SECTION ========================= #
    gamma = 0.9
    reward = 10.0
    period = 5
    offset = 0  # offset=0 means R1 is the first reward; offset=4 means first reward is R5
    decimals = 3
    # ==================================================================== #

    G0 = periodic_return(gamma=gamma, reward=reward, period=period, offset=offset)

    print(f"gamma   = {gamma}")
    print(f"reward  = {reward}")
    print(f"period  = {period}")
    print(f"offset  = {offset}")
    print()
    print(f"G0 = {G0:.{decimals}f}")


if __name__ == "__main__":
    main()

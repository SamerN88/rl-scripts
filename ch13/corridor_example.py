"""
Example 13.1 Short corridor with switched actions (p. 323)
"""

import time
import numpy as np
import matplotlib.pyplot as plt


def trust(p):
    return np.random.random() < p


def run_episode(p_trust):
    p = p_trust

    if np.isclose(p, 0) or np.isclose(p, 1):
        return -np.inf

    ret = 0
    s = 1
    while s != 'G':
        if s == 1:
            s = 2 if trust(p) else 1
        elif s == 2:
            s = 1 if trust(p) else 3
        elif s == 3:
            s = 'G' if trust(p) else 2

        ret -= 1

    return ret


def mc_avg_return(p_trust, n):
    return sum(run_episode(p_trust) for _ in range(n)) / n


def main():
    start_time = time.time()

    # TODO: These determine the accuracy/stability of the results; tune as needed.
    p_resolution = 500  # accuracy
    mc_trials = 20000  # stability

    print('Run config:')
    print(f'    p_resolution = {p_resolution}')
    print(f'    mc_trials = {mc_trials}')
    print('-' * 75)
    print()

    runs = []

    best_p = None
    best_ret = -np.inf
    for p_trust in np.linspace(0, 1, p_resolution):
        avg_ret = mc_avg_return(p_trust, mc_trials)

        runs.append((p_trust, avg_ret))

        if avg_ret > best_ret:
            best_ret = avg_ret
            best_p = p_trust

        print(f'p_trust = {p_trust}')
        print(f'avg return = {avg_ret}')
        print()

    print('-' * 75)
    print(f'optimal p_trust = {best_p}')
    print(f'optimal return = {best_ret}')

    print()
    print(f'Runtime: {time.time() - start_time:.3f}s')

    runs = np.array(runs)
    plt.plot(runs[:, 0], runs[:, 1])
    plt.plot(best_p, best_ret, color='red', marker='o')
    plt.show()


if __name__ == '__main__':
    main()

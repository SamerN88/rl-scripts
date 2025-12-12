import numpy as np
import random
from collections import defaultdict

import time
try:
    from tqdm import trange  # optional, nice progress bar
except Exception:
    trange = None

A_INCREMENTS = [(ax, ay) for ax in (-1,0,1) for ay in (-1,0,1)]
V_MAX = 5
NOISE_P = 0#0.1


def bresenham_line(x0, y0, x1, y1):
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    cells = [(x, y)]
    while (x, y) != (x1, y1):
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
        cells.append((x, y))
    return cells


class Racetrack:
    def __init__(self, ascii_map):
        lines = [list(row.rstrip('\n')) for row in ascii_map.strip('\n').split('\n')]
        self.H = len(lines)
        self.W = max(len(r) for r in lines)
        for i in range(len(lines)):
            if len(lines[i]) < self.W:
                lines[i] += [' '] * (self.W - len(lines[i]))
        self.grid = np.array(lines)
        self.start_cells = [(x,y) for y in range(self.H) for x in range(self.W) if self.grid[y,x]=='S']
        self.finish_cells = {(x,y) for y in range(self.H) for x in range(self.W) if self.grid[y,x]=='F'}

    def in_bounds(self, x, y):
        return 0 <= x < self.W and 0 <= y < self.H

    def is_track(self, x, y):
        return self.in_bounds(x,y) and (self.grid[y,x] in {'.', 'S', 'F'})

    def random_start(self):
        s = random.choice(self.start_cells)
        return (s, (0,0))

    def valid_actions(self, state):
        (x, y), (vx, vy) = state
        valid = []
        for i, (ax, ay) in enumerate(A_INCREMENTS):
            nvx = max(0, min(V_MAX, vx + ax))
            nvy = max(0, min(V_MAX, vy + ay))
            # Disallow staying at (0,0) (prevents stalling)
            if nvx == 0 and nvy == 0:
                continue
            valid.append(i)
        return valid

    def segment_hit(self, x, y, vx, vy):
        # positive vy means moving UP (toward smaller y indices)
        nx, ny = x + vx, y - vy
        cells = bresenham_line(x, y, nx, ny)
        for cx, cy in cells[1:]:
            if (cx,cy) in self.finish_cells:
                return 'finish', cx, cy
            if not self.is_track(cx,cy):
                return 'crash', x, y
        return 'ok', nx, ny

    def step(self, state, action, noise=True):
        (x,y),(vx,vy) = state
        ax, ay = A_INCREMENTS[action]
        if noise and random.random() < NOISE_P:
            ax, ay = 0, 0
        nvx = max(0, min(V_MAX, vx+ax))
        nvy = max(0, min(V_MAX, vy+ay))
        # disallow zero-velocity off start line
        if nvx == 0 and nvy == 0 and self.grid[y,x] != 'S':
            nvx, nvy = vx, vy
        outcome, nx, ny = self.segment_hit(x,y,nvx,nvy)
        if outcome=='finish':
            return ((nx,ny),(nvx,nvy)), -1, True, False
        elif outcome=='crash':
            (sx,sy),(vv) = self.random_start()
            return ((sx,sy),(0,0)), -1, False, True
        else:
            return ((nx,ny),(nvx,nvy)), -1, False, False


def _simple_progress(ep, total, start_time, width=50):
    # ep is 0-based; display 1-based for humans
    done = ep + 1
    pct = done / total
    filled = int(pct * width)
    bar = '█' * filled + '·' * (width - filled)
    elapsed = time.perf_counter() - start_time
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else float('inf')
    print(f"\rTraining [{bar}] {pct*100:5.1f}% | {done}/{total} | "
          f"elapsed {elapsed:6.1f}s | ETA {eta:6.1f}s", end='', flush=True)


def mc_control(env, num_episodes=10000, epsilon=0.1, gamma=1.0, max_steps=1000, seed=0, show_progress=True):
    random.seed(seed)
    np.random.seed(seed)

    Q = defaultdict(lambda: np.zeros(len(A_INCREMENTS)))
    C = defaultdict(lambda: np.zeros(len(A_INCREMENTS)))
    policy = {}

    # Choose progress iterator
    if show_progress and trange is not None:
        iterator = trange(num_episodes, desc="Training (MC control)", ncols=80, leave=False)
        use_simple = False
    else:
        iterator = range(num_episodes)
        use_simple = show_progress
        start_time = time.perf_counter()

    for ep in iterator:
        state = env.random_start()
        episode = []
        for t in range(max_steps):
            valid = env.valid_actions(state)
            if random.random() < epsilon:
                a = random.choice(valid)
            else:
                mask = np.full(len(A_INCREMENTS), -1e9)
                mask[valid] = 0.0
                a = int(np.argmax(Q[state] + mask))

            next_state, r, done, _ = env.step(state, a, noise=True)
            episode.append((state, a, r))
            state = next_state
            if done:
                break

        # First-visit updates
        G = 0.0
        seen = set()
        for (s,a,r) in reversed(episode):
            G = gamma*G + r
            if (s,a) not in seen:
                seen.add((s,a))
                C[s][a] += 1.0
                alpha = 1.0 / C[s][a]
                Q[s][a] += alpha * (G - Q[s][a])

        for (s, _, _) in episode:
            valid = env.valid_actions(s)
            mask = np.full(len(A_INCREMENTS), -1e9)
            mask[valid] = 0.0
            policy[s] = int(np.argmax(Q[s] + mask))

        # Update simple progress bar occasionally
        if use_simple:
            # update ~100 times across training (every 1%)
            if (ep + 1) % max(1, num_episodes // 100) == 0 or (ep + 1) == num_episodes:
                _simple_progress(ep, num_episodes, start_time)

    if show_progress and trange is None:
        print()  # newline after the simple progress bar

    return Q, policy


# Helpers for rollout/eval

def greedy_action(env, Q, state):
    """Greedy action from Q, masked to valid actions; works even if state unseen."""
    valid = env.valid_actions(state)
    q = Q[state]  # zeros for unseen states (defaultdict)
    mask = np.full(len(q), -1e9)
    mask[valid] = 0.0
    return int(np.argmax(q + mask))


def evaluate(env, Q, n=100, max_steps=500):
    """Average steps to finish (noise off) from random starts."""
    total = []
    for _ in range(n):
        state = env.random_start()
        done = False
        steps = 0
        while not done and steps < max_steps:
            # prefer learned policy if present; otherwise greedy from Q
            a = greedy_action(env, Q, state)  # instead of policy.get(...)
            state, r, done, crashed = env.step(state, a, noise=False)
            steps += 1
        total.append(steps)
    return float(np.mean(total)), float(np.std(total))


def render_with_path(env, path_xy):
    """
    Return a string of the track with the car path drawn as '*'.
    Keeps 'F' visible on finish cells and 'S' on start cells.
    """
    canvas = [row.tolist() for row in env.grid]  # deep copy to lists

    for i, (x, y) in enumerate(path_xy):
        if not env.in_bounds(x, y):
            continue

        if i == 0:
            canvas[y][x] = '$'
        elif i == len(path_xy) - 1:
            canvas[y][x] = '#'
        else:
            canvas[y][x] = '*'

    # join back to lines
    return "\n".join("".join(row).rstrip() for row in canvas)


def main():
    # Tiny right-turn for quick demo
    track_str = """
             ............        FFFFFFFFFFF    
        ....................................
       ...................................
      ..................................
     ...........        ............
    SSSSSSSSSSS            
    """

    env = Racetrack(track_str)
    num_episodes = 10000
    max_steps = 200
    epsilon = 0.1  # noise probability
    gamma = 1  # tweak as needed

    print('Track:')
    print(track_str)
    print()
    print('Training config:')
    print(f'    total episodes: {num_episodes:,}')
    print(f'    max steps per episode: {max_steps:,}')
    print(f'    epsilon: {epsilon}')
    print(f'    gamma: {gamma}')
    print('_' * 50)
    print()

    # Train (increase num_episodes for harder/bigger tracks)
    Q, policy = mc_control(env, num_episodes=num_episodes, epsilon=epsilon, gamma=gamma)

    print('RESULTS:\n')
    # Evaluate many starts with noise off
    avg, std = evaluate(env, Q, n=200, max_steps=max_steps)
    if avg == max_steps and std == 0:
        print('FAILED to learn winning policy.')
    else:
        print(f"Avg steps to finish (noise off): {avg:.1f} ± {std:.1f}")

    # One illustrative rollout (noise off)
    state = env.random_start()
    done = False
    steps = 0
    path_xy = []
    print("\nSample rollout (noise off):")
    while not done and steps < max_steps:
        a = greedy_action(env, Q, state)  # instead of policy.get(...)
        (pos, vel) = state
        path_xy.append(pos)  # record current cell (x, y)
        state, r, done, crashed = env.step(state, a, noise=False)
        steps += 1
        print(f"Step {steps:3d}   |   pos={pos}  vel={vel}  action={a}  reward={r}")

    # Also record the terminal position if finished in-bounds
    path_xy.append(state[0])

    print()
    if done:
        print(f'🎉Finished track in {steps} steps!')
    else:
        print(f'😭FAILED TO FINISH (max {max_steps} steps).')

    # Render the track with the path
    print("\n\nPath on track ('$' is start, '#' is end, '*' is path in between):\n")
    print(render_with_path(env, path_xy))


if __name__ == '__main__':
    main()

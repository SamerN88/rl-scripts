# Monte Carlo soap-bubble surface (Laplace) on a square frame
# - We fix heights along the boundary (the "wire frame")
# - For each interior grid point we launch many random walks that stop at the boundary
# - The estimate of the surface at that point is the average boundary height where the walks exit
#
# This is a direct implementation of the Monte Carlo method described in the prompt image.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3D projection
import time

# -----------------------------------------------------------
# Parameters (N=20 with num_walks=500 takes about 15 sec)
# -----------------------------------------------------------
N = 20               # grid size (NxN); includes boundary
num_walks = 500      # number of random walks per interior grid point
rng = np.random.default_rng(0)

# -----------------------------------------------------------
# Make a fixed boundary ("wire frame")
# -----------------------------------------------------------
# Domain: [0, 1] x [0, 1] square
xs = np.linspace(0.0, 1.0, N)
ys = np.linspace(0.0, 1.0, N)
X, Y = np.meshgrid(xs, ys, indexing="ij")

# Boundary values z(x, y)
# - three edges at height 0
# - the top edge shaped with a bump and ripple (nontrivial surface)
boundary = np.zeros((N, N))

# top edge (y = 1): a smooth bump centered at x=0.6 with a small ripple
top = 1.2 * np.exp(-((xs - 0.6) ** 2) / 0.01) + 0.25 * np.sin(4 * np.pi * xs)
boundary[:, -1] = top

# also add a small "hook" on the left edge (x = 0)
boundary[0, :] += 0.6 * np.exp(-((ys - 0.35) ** 2) / 0.015)

# Ensure corners are consistent (average the contributions if any)
boundary[0, -1] = (top[0] + boundary[0, -1]) / 2.0
boundary[-1, -1] = top[-1]  # bottom of right column uses only top contribution

# -----------------------------------------------------------
# Monte Carlo estimator
# -----------------------------------------------------------
is_boundary = np.zeros((N, N), dtype=bool)
is_boundary[0, :] = True
is_boundary[-1, :] = True
is_boundary[:, 0] = True
is_boundary[:, -1] = True

Z = np.zeros((N, N))
Z[is_boundary] = boundary[is_boundary]

start_time = time.time()

# Helper: perform a single random neighbor step (4-neighborhood)
moves = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=int)

# For each interior grid point, launch random walks
for i in range(1, N - 1):
    for j in range(1, N - 1):
        total = 0.0
        for _ in range(num_walks):
            x, y = i, j
            while not is_boundary[x, y]:
                dx, dy = moves[rng.integers(0, 4)]
                nx, ny = x + dx, y + dy
                # reflect if step tries to go outside array (shouldn't happen for interior, but safe)
                if nx < 0 or nx >= N or ny < 0 or ny >= N:
                    continue
                x, y = nx, ny
            total += boundary[x, y]
        Z[i, j] = total / num_walks

elapsed = time.time() - start_time
print(f"Computed Monte Carlo surface on {N}x{N} grid with {num_walks} walks per point "
      f"in {elapsed:.2f} seconds.")

# -----------------------------------------------------------
# Plot the estimated surface
# -----------------------------------------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True)
ax.set_title("Monte Carlo Soap-Bubble Surface (estimated harmonic function)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("height")
plt.tight_layout()

plt.show()

# Save image for download
# out_path = "soap_bubble_surface.png"
# plt.savefig(out_path, dpi=160, bbox_inches="tight")
# print(f"Saved plot to: {out_path}")

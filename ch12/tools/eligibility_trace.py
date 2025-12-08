import numpy as np


def compute_eligibility_trace(states, feature_fn, *, gamma, lam, until_index=None, z_init=None):
    """
    Compute the eligibility trace vector z_t by iterating:
        z_{-1} = 0
        z_t = gamma * lam * z_{t-1} + feature_fn(S_t)

    Args:
        states: iterable of states in the trajectory [S_0, S_1, ..., S_T]
        feature_fn: callable s -> feature/gradient vector (numpy array, shape [d])
        gamma: discount factor (float)
        lam: trace decay (float)
        until_index: if not None, stop after processing states[until_index] (inclusive)
        z_init: optional initial z_{-1}; if None uses zeros of correct dim

    Returns:
        z: numpy array with the eligibility trace after the last processed state
    """
    if not states:
        raise ValueError("states must be non-empty")

    d = len(feature_fn(states[0]))
    z = np.zeros(d, dtype=float) if z_init is None else np.array(z_init, dtype=float)

    last = len(states) - 1 if until_index is None else int(until_index)
    for t in range(0, last + 1):
        z = gamma * lam * z + feature_fn(states[t])
    return z


def make_one_hot_feature(index_map, n_features):
    """
    Build a one-hot feature function for tabular problems.

    Args:
        index_map: dict mapping state labels -> integer index in [0, n_features-1]
        n_features: total number of features

    Returns:
        feature_fn(s): one-hot vector with 1 at index_map[s]
    """
    eye = np.eye(n_features, dtype=float)
    def feature_fn(s):
        return eye[index_map[s]]
    return feature_fn


def main():
    # --- Example from Canvas HW (6 one-hot features: [G, 1, 2, 3, 4, 5]) ---
    index_map = {'G': 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

    n_features = len(index_map)
    feature_fn = make_one_hot_feature(index_map, n_features)

    gamma = 0.9
    lam = 0.3

    # Trajectory segment (we only care up to after processing state 3 here)
    states = [5, 4, 3, 2, 1, 'G']

    # Compute z_t after processing state 3 (i.e., after states[0], states[1], states[2])
    z_after_s3 = compute_eligibility_trace(
        states, feature_fn, gamma=gamma, lam=lam, until_index=index_map[3]  # until state 3
    )

    print("Eligibility trace after processing state 3:")
    print(z_after_s3)

    # Value of the component associated with state 5 (the 'last' component)
    comp_state5 = z_after_s3[index_map[5]]
    print(f"Component for state 5 after state 3 has been added: {comp_state5:.6f}")


if __name__ == "__main__":
    main()

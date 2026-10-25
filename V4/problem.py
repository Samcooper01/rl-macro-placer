"""V3 problem GENERATOR: a fresh random problem every episode.

V1/V2 trained on one fixed problem, so the agent could memorize it. V3 draws a
new random problem each episode (random macro sizes and a random netlist), so
the agent must learn a general placement policy that works on unseen problems.
Object counts are fixed (so observation and action spaces stay fixed); only the
sizes and the connectivity change.
"""

import numpy as np

CANVAS = 25
NUM_MACROS = 4
NUM_CLUSTERS = 12
SIZE_MIN, SIZE_MAX = 3, 5      # macro side length range
AREA_MIN, AREA_MAX = 3, 6      # cluster area range

M = NUM_MACROS
N = NUM_MACROS + NUM_CLUSTERS


def generate_problem(rng):
    """Return (macro_sizes, cluster_areas, nets) for one random problem."""
    sizes = [(int(rng.integers(SIZE_MIN, SIZE_MAX + 1)),
              int(rng.integers(SIZE_MIN, SIZE_MAX + 1))) for _ in range(NUM_MACROS)]
    areas = [int(rng.integers(AREA_MIN, AREA_MAX + 1)) for _ in range(NUM_CLUSTERS)]

    nets = []
    # each cluster connects to 1-2 random other objects
    for c in range(M, N):
        k = int(rng.integers(1, 3))
        partners = rng.choice([o for o in range(N) if o != c], size=k, replace=False)
        nets.append([c] + list(map(int, partners)))
    # a few wider structural nets
    for _ in range(3):
        k = int(rng.integers(2, 4))
        nets.append(list(map(int, rng.choice(N, size=k, replace=False))))
    # make sure every macro appears in at least one net
    used = {o for net in nets for o in net}
    for m in range(M):
        if m not in used:
            nets.append([m, int(rng.integers(M, N))])
    return sizes, areas, nets

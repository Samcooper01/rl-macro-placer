"""Scale-parameterized problem generator for the scaling study.

A 'scale' is (num_macros, num_clusters, canvas). Larger scales are harder
placement problems with more structure, which is where a graph encoder should
help if the paper's advantage is real.
"""
import numpy as np

SIZE_MIN = 3


def size_max(canvas):
    return max(SIZE_MIN + 1, canvas // 6)


def generate_problem(rng, M, C, canvas):
    smax = size_max(canvas)
    sizes = [(int(rng.integers(SIZE_MIN, smax + 1)),
              int(rng.integers(SIZE_MIN, smax + 1))) for _ in range(M)]
    areas = [int(rng.integers(3, 7)) for _ in range(C)]
    N = M + C
    nets = []
    for c in range(M, N):
        k = int(rng.integers(1, 3))
        partners = rng.choice([o for o in range(N) if o != c], size=k, replace=False)
        nets.append([c] + list(map(int, partners)))
    for _ in range(max(2, M)):
        k = int(rng.integers(2, 4))
        nets.append(list(map(int, rng.choice(N, size=k, replace=False))))
    used = {o for net in nets for o in net}
    for m in range(M):
        if m not in used:
            nets.append([m, int(rng.integers(M, N))])
    return sizes, areas, nets

"""Force-directed placement of the soft clusters around the fixed macros.

Given the macro positions the RL agent chose (fixed anchors), place the clusters
by iterating a spring-electrical model:
  - attractive (spring) force: for each net, pull its objects toward the net centroid
  - repulsive (electrical) force: all objects push each other apart (spreads clusters)
Macros do not move; clusters do. Bounded iterations keep it cheap enough to run
inside every RL episode.
"""

import numpy as np

K_ATT = 0.60     # spring strength (pull connected objects together)
K_REP = 1.5      # repulsion strength (mild: just avoid piling)
STEP = 0.30      # integration step
ITERS = 80       # iterations per placement (bounded for speed)
K_BLOCK = 4.0    # push clusters out of macro rectangles (macros are keep-outs)


def force_directed_place(macro_pos, macro_sizes, num_macros, num_clusters, nets,
                         canvas, iters=ITERS, seed=None):
    rng = np.random.default_rng(seed)
    N = num_macros + num_clusters
    pos = np.zeros((N, 2), dtype=np.float64)
    pos[:num_macros] = np.asarray(macro_pos, dtype=np.float64)
    # initialise clusters near the centre with a little jitter
    pos[num_macros:] = canvas / 2.0 + rng.standard_normal((num_clusters, 2)) * 2.0

    movable = slice(num_macros, N)

    for _ in range(iters):
        force = np.zeros((N, 2))

        # attractive: star model per net (pull toward the net's centroid)
        for net in nets:
            idx = np.asarray(net)
            centroid = pos[idx].mean(axis=0)
            force[idx] += K_ATT * (centroid - pos[idx])

        # repulsive: pairwise, vectorised
        diff = pos[:, None, :] - pos[None, :, :]        # (N, N, 2)
        dist2 = (diff ** 2).sum(-1) + 1.0               # softened
        rep = K_REP * diff / dist2[:, :, None]
        force += rep.sum(axis=1)

        # blockage: macros are keep-outs, push any cluster that is inside a
        # macro rectangle out along its shallowest penetration axis.
        for m in range(num_macros):
            mx, my = pos[m]
            w, h = macro_sizes[m]
            hx, hy = w / 2.0 + 0.5, h / 2.0 + 0.5
            for k in range(num_macros, N):
                dx, dy = pos[k, 0] - mx, pos[k, 1] - my
                if abs(dx) < hx and abs(dy) < hy:
                    px, py = hx - abs(dx), hy - abs(dy)   # penetration depths
                    if px < py:
                        force[k, 0] += K_BLOCK * (1.0 if dx >= 0 else -1.0) * px
                    else:
                        force[k, 1] += K_BLOCK * (1.0 if dy >= 0 else -1.0) * py

        pos[movable] += STEP * force[movable]
        pos[movable] = np.clip(pos[movable], 0, canvas)

    # final legalization: hard-push any cluster still inside a macro to the edge
    for m in range(num_macros):
        mx, my = pos[m]
        w, h = macro_sizes[m]
        hx, hy = w / 2.0 + 0.4, h / 2.0 + 0.4
        for k in range(num_macros, N):
            dx, dy = pos[k, 0] - mx, pos[k, 1] - my
            if abs(dx) < hx and abs(dy) < hy:
                if hx - abs(dx) < hy - abs(dy):
                    pos[k, 0] = mx + (hx if dx >= 0 else -hx)
                else:
                    pos[k, 1] = my + (hy if dy >= 0 else -hy)
    pos[num_macros:] = np.clip(pos[num_macros:], 0, canvas)

    return pos

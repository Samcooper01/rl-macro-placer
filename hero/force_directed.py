"""Force-directed placement of the soft clusters around the fixed macros.

Springs pull each object toward the centroid of every net it is in, and a
SHORT-RANGE repulsion spreads apart only objects that are close (so clusters
nestle among and around the macros instead of being flung to the canvas edges).
Macros are keep-out blockages and do not move. Bounded iterations keep it cheap
enough to run inside every RL episode.
"""

import numpy as np

K_ATT = 1.0      # spring strength (pull connected objects together)
K_REP = 3.0      # short-range repulsion strength
R_REP = 3.5      # repulsion radius: only objects closer than this push apart
STEP = 0.30      # integration step
ITERS = 60       # iterations per placement (bounded for speed)
K_BLOCK = 4.0    # push clusters out of macro rectangles (macros are keep-outs)
GAP = 1.0        # keep-out margin beyond the macro edge (clears the cluster's drawn radius)


def force_directed_place(macro_pos, macro_sizes, num_macros, num_clusters, nets,
                         canvas, iters=ITERS, seed=0):
    rng = np.random.default_rng(seed)
    N = num_macros + num_clusters
    pos = np.zeros((N, 2), dtype=np.float64)
    pos[:num_macros] = np.asarray(macro_pos, dtype=np.float64)
    # start clusters near the macro group so they settle among the macros
    center = pos[:num_macros].mean(axis=0)
    pos[num_macros:] = center + rng.standard_normal((num_clusters, 2)) * 1.5
    mv = slice(num_macros, N)

    for _ in range(iters):
        force = np.zeros((N, 2))

        # attractive: star model per net (pull toward the net centroid)
        for net in nets:
            idx = np.asarray(net)
            force[idx] += K_ATT * (pos[idx].mean(axis=0) - pos[idx])

        # short-range repulsion: only objects within R_REP push apart
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1)) + 1e-6
        close = (dist < R_REP) & (dist > 1e-3)
        push = np.where(close[:, :, None],
                        K_REP * ((R_REP - dist) / R_REP)[:, :, None] * diff / dist[:, :, None],
                        0.0)
        force += push.sum(axis=1)

        # blockage: push clusters out of macro rectangles
        for m in range(num_macros):
            mx, my = pos[m]
            w, h = macro_sizes[m]
            hx, hy = w / 2.0 + GAP, h / 2.0 + GAP
            for k in range(num_macros, N):
                dx, dy = pos[k, 0] - mx, pos[k, 1] - my
                if abs(dx) < hx and abs(dy) < hy:
                    px, py = hx - abs(dx), hy - abs(dy)
                    if px < py:
                        force[k, 0] += K_BLOCK * (1.0 if dx >= 0 else -1.0) * px
                    else:
                        force[k, 1] += K_BLOCK * (1.0 if dy >= 0 else -1.0) * py

        pos[mv] += STEP * force[mv]
        pos[mv] = np.clip(pos[mv], 0, canvas)

    # final legalization: iterate to a fixed point so a cluster pushed out of one
    # macro cannot stay lodged inside another. Keep-out clears the drawn radius.
    for _ in range(40):
        moved = False
        for m in range(num_macros):
            mx, my = pos[m]
            w, h = macro_sizes[m]
            hx, hy = w / 2.0 + GAP, h / 2.0 + GAP
            for k in range(num_macros, N):
                dx, dy = pos[k, 0] - mx, pos[k, 1] - my
                if abs(dx) < hx and abs(dy) < hy:
                    if hx - abs(dx) < hy - abs(dy):
                        pos[k, 0] = mx + (hx if dx >= 0 else -hx)
                    else:
                        pos[k, 1] = my + (hy if dy >= 0 else -hy)
                    moved = True
        pos[num_macros:] = np.clip(pos[num_macros:], 0, canvas)
        if not moved:
            break
    return pos

"""Wirelength, congestion, and density estimators over a full placement.

All three are cheap proxies computed from object positions (and sizes), so they
can run inside every RL episode. Lower is better for all three.
"""

import numpy as np


def hpwl(pos, nets):
    """Sum over nets of the bounding-box half-perimeter."""
    total = 0.0
    for net in nets:
        p = pos[np.asarray(net)]
        total += (p[:, 0].max() - p[:, 0].min()) + (p[:, 1].max() - p[:, 1].min())
    return total


def congestion(pos, nets, canvas, res=25, top_frac=0.10):
    """RUDY-style routing-demand estimate.

    Each net spreads its wirelength demand uniformly over the cells of its
    bounding box. Congestion is the mean of the most-congested cells.
    """
    grid = np.zeros((res, res))
    scale = res / canvas
    for net in nets:
        p = pos[np.asarray(net)]
        x0, x1 = p[:, 0].min(), p[:, 0].max()
        y0, y1 = p[:, 1].min(), p[:, 1].max()
        demand = (x1 - x0) + (y1 - y0)                  # net HPWL as its demand
        c0, c1 = int(x0 * scale), int(np.ceil(x1 * scale))
        r0, r1 = int(y0 * scale), int(np.ceil(y1 * scale))
        c1 = max(c1, c0 + 1); r1 = max(r1, r0 + 1)
        c0, c1 = np.clip([c0, c1], 0, res); r0, r1 = np.clip([r0, r1], 0, res)
        area = max((c1 - c0) * (r1 - r0), 1)
        grid[r0:r1, c0:c1] += demand / area
    flat = np.sort(grid.ravel())[::-1]
    k = max(1, int(len(flat) * top_frac))
    return float(flat[:k].mean())


def density(pos, areas, canvas, res=8, top_frac=0.15):
    """Placement-density estimate: object area rasterised onto a grid.

    Density is the mean of the most-occupied cells (encourages spreading area out).
    """
    grid = np.zeros((res, res))
    scale = res / canvas
    for i, a in enumerate(areas):
        c = int(np.clip(pos[i, 0] * scale, 0, res - 1))
        r = int(np.clip(pos[i, 1] * scale, 0, res - 1))
        grid[r, c] += a
    flat = np.sort(grid.ravel())[::-1]
    k = max(1, int(len(flat) * top_frac))
    return float(flat[:k].mean())

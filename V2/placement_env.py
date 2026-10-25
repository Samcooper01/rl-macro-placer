"""V2 environment: RL places sized macros, then a force-directed pass places the
standard-cell clusters, and the reward is a 3-term proxy over the full placement
(wirelength + congestion + density), as in the Google graph-placement method.

Macros are placed sequentially by lower-left corner on the grid; action masking
forbids any corner whose footprint would overlap an existing macro or leave the
canvas. When all macros are placed, force_directed_place() positions the clusters
and metrics.py scores the whole design.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import problem
import metrics
from force_directed import force_directed_place


class PlacementEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    W_WL, W_CONG, W_DENS = 0.1, 10.0, 2.0   # reward weights (balance the 3 terms)

    def __init__(self):
        super().__init__()
        self.canvas = problem.CANVAS
        self.M = problem.NUM_MACROS
        self.C = problem.NUM_CLUSTERS
        self.sizes = problem.MACRO_SIZES
        self.nets = problem.NETS
        self.areas = ([float(w * h) for (w, h) in self.sizes]
                      + [float(a) for a in problem.CLUSTER_AREAS])
        self.ncells = self.canvas * self.canvas

        self.action_space = spaces.Discrete(self.ncells)
        obs_len = self.ncells + self.ncells + self.M
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_len,), dtype=np.float32)

        # macro-macro adjacency: two macros are neighbors if they share a net.
        self.macro_adj = [set() for _ in range(self.M)]
        for net in self.nets:
            ms = [o for o in net if o < self.M]
            for a in ms:
                for b in ms:
                    if a != b:
                        self.macro_adj[a].add(b)

        self.grid = None
        self.corners = None      # corners[i] = (row, col) lower-left of macro i
        self.next_macro = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((self.canvas, self.canvas), dtype=np.float32)
        self.corners = [None] * self.M
        self.next_macro = 0
        return self._get_obs(), {}

    def action_masks(self):
        """A corner is legal if the current macro's footprint fits on the canvas
        and does not overlap any already-placed macro."""
        w, h = self.sizes[self.next_macro] if self.next_macro < self.M else (1, 1)
        mask = np.zeros(self.ncells, dtype=bool)
        for r in range(self.canvas - h + 1):
            row_free = self.grid[r:r + h]
            for c in range(self.canvas - w + 1):
                if row_free[:, c:c + w].sum() == 0:
                    mask[r * self.canvas + c] = True
        return mask

    def step(self, action):
        r, c = divmod(int(action), self.canvas)
        w, h = self.sizes[self.next_macro]
        self.grid[r:r + h, c:c + w] = 1.0
        self.corners[self.next_macro] = (r, c)
        self.next_macro += 1

        terminated = self.next_macro >= self.M
        reward = -self._full_cost() if terminated else 0.0
        return self._get_obs(), reward, terminated, False, {}

    def _macro_centers(self):
        cen = np.zeros((self.M, 2))
        for i, (r, c) in enumerate(self.corners):
            w, h = self.sizes[i]
            cen[i] = (c + w / 2.0, r + h / 2.0)
        return cen

    def full_placement(self, seed=0):
        """Place clusters around the fixed macros and return all object positions."""
        macro_pos = self._macro_centers()
        return force_directed_place(macro_pos, self.sizes, self.M, self.C,
                                    self.nets, self.canvas, seed=seed)

    def _full_cost(self):
        pos = self.full_placement(seed=0)   # fixed seed -> deterministic reward
        wl = metrics.hpwl(pos, self.nets)
        cg = metrics.congestion(pos, self.nets, self.canvas)
        dn = metrics.density(pos, self.areas, self.canvas)
        return self.W_WL * wl + self.W_CONG * cg + self.W_DENS * dn

    def cost_terms(self):
        pos = self.full_placement(seed=0)
        return (metrics.hpwl(pos, self.nets),
                metrics.congestion(pos, self.nets, self.canvas),
                metrics.density(pos, self.areas, self.canvas))

    def _get_obs(self):
        occ = self.grid.flatten()
        neigh = np.zeros((self.canvas, self.canvas), dtype=np.float32)
        onehot = np.zeros(self.M, dtype=np.float32)
        if self.next_macro < self.M:
            onehot[self.next_macro] = 1.0
            for j in self.macro_adj[self.next_macro]:
                if self.corners[j] is not None:
                    r, c = self.corners[j]
                    w, h = self.sizes[j]
                    neigh[r:r + h, c:c + w] = 1.0
        return np.concatenate([occ, neigh.flatten(), onehot]).astype(np.float32)

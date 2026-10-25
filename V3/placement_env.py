"""V3 environment: same pipeline as V2 (RL places sized macros, force-directed
places the clusters, 3-term proxy reward), but a NEW RANDOM PROBLEM every episode
so the agent must learn a general policy instead of memorizing one layout.

reset(options={"problem": (sizes, areas, nets)}) injects a specific problem (used
for held-out evaluation); otherwise a fresh random problem is drawn.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import problem
import metrics
from force_directed import force_directed_place


class PlacementEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    W_WL, W_CONG, W_DENS = 0.1, 10.0, 2.0

    def __init__(self, gen_seed=0):
        super().__init__()
        self.canvas = problem.CANVAS
        self.M = problem.NUM_MACROS
        self.C = problem.NUM_CLUSTERS
        self.ncells = self.canvas * self.canvas
        self._gen_rng = np.random.default_rng(gen_seed)

        self.action_space = spaces.Discrete(self.ncells)
        # obs: occupancy | neighbor-positions | one-hot(next macro) | current size (2)
        obs_len = self.ncells + self.ncells + self.M + 2
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_len,), dtype=np.float32)

        # per-episode problem state:
        self.sizes = self.cluster_areas = self.nets = None
        self.areas = self.macro_adj = None
        self.grid = self.corners = self.next_macro = None

    def _set_problem(self, prob):
        self.sizes, self.cluster_areas, self.nets = prob
        self.areas = ([float(w * h) for (w, h) in self.sizes]
                      + [float(a) for a in self.cluster_areas])
        self.macro_adj = [set() for _ in range(self.M)]
        for net in self.nets:
            ms = [o for o in net if o < self.M]
            for a in ms:
                for b in ms:
                    if a != b:
                        self.macro_adj[a].add(b)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and "problem" in options:
            self._set_problem(options["problem"])
        else:
            self._set_problem(problem.generate_problem(self._gen_rng))
        self.grid = np.zeros((self.canvas, self.canvas), dtype=np.float32)
        self.corners = [None] * self.M
        self.next_macro = 0
        return self._get_obs(), {}

    def action_masks(self):
        w, h = self.sizes[self.next_macro] if self.next_macro < self.M else (1, 1)
        n = self.canvas
        S = np.zeros((n + 1, n + 1))
        S[1:, 1:] = np.cumsum(np.cumsum(self.grid, axis=0), axis=1)
        R, C = n - h + 1, n - w + 1
        # footprint sum for every legal corner via the summed-area table
        sums = (S[h:h + R, w:w + C] - S[0:R, w:w + C]
                - S[h:h + R, 0:C] + S[0:R, 0:C])
        mask = np.zeros((n, n), dtype=bool)
        mask[:R, :C] = (sums == 0)
        return mask.flatten()

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
        return force_directed_place(self._macro_centers(), self.sizes, self.M,
                                    self.C, self.nets, self.canvas, seed=seed)

    def _full_cost(self):
        pos = self.full_placement(seed=0)
        return (self.W_WL * metrics.hpwl(pos, self.nets)
                + self.W_CONG * metrics.congestion(pos, self.nets, self.canvas)
                + self.W_DENS * metrics.density(pos, self.areas, self.canvas))

    def cost_terms(self):
        pos = self.full_placement(seed=0)
        return (metrics.hpwl(pos, self.nets),
                metrics.congestion(pos, self.nets, self.canvas),
                metrics.density(pos, self.areas, self.canvas))

    def _get_obs(self):
        occ = self.grid.flatten()
        neigh = np.zeros((self.canvas, self.canvas), dtype=np.float32)
        onehot = np.zeros(self.M, dtype=np.float32)
        size = np.zeros(2, dtype=np.float32)
        if self.next_macro < self.M:
            onehot[self.next_macro] = 1.0
            w, h = self.sizes[self.next_macro]
            size[:] = (w / problem.SIZE_MAX, h / problem.SIZE_MAX)
            for j in self.macro_adj[self.next_macro]:
                if self.corners[j] is not None:
                    r, c = self.corners[j]
                    wj, hj = self.sizes[j]
                    neigh[r:r + hj, c:c + wj] = 1.0
        return np.concatenate([occ, neigh.flatten(), onehot, size]).astype(np.float32)

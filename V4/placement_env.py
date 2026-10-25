"""V4 environment: same pipeline as V3 (RL places sized macros, force-directed
places the clusters, 3-term proxy reward, random problem each episode), but the
observation is now a GRAPH so a GNN (Graph Neural Network) policy can read the
netlist structure directly:

  nodes: per-object feature matrix (macros and clusters)
  adj:   object-object adjacency (two objects are adjacent if they share a net)
  grid:  the macro-occupancy grid (for spatial placement and action masking)

This is the piece that lets the policy generalize across netlist structures,
which the flat MLP (Multi-Layer Perceptron) of V3 could not represent.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import problem
import metrics
from force_directed import force_directed_place

NODE_FEATS = 8   # [is_macro, is_cluster, is_placed, is_current, sizeA, sizeB, x, y]


class PlacementEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    W_WL, W_CONG, W_DENS = 0.1, 10.0, 2.0

    def __init__(self, gen_seed=0):
        super().__init__()
        self.canvas = problem.CANVAS
        self.M = problem.NUM_MACROS
        self.C = problem.NUM_CLUSTERS
        self.N = self.M + self.C
        self.ncells = self.canvas * self.canvas
        self._gen_rng = np.random.default_rng(gen_seed)

        self.action_space = spaces.Discrete(self.ncells)
        self.observation_space = spaces.Dict({
            "nodes": spaces.Box(-1.0, 1.0, shape=(self.N, NODE_FEATS), dtype=np.float32),
            "adj": spaces.Box(0.0, 1.0, shape=(self.N, self.N), dtype=np.float32),
            "grid": spaces.Box(0.0, 1.0, shape=(self.canvas, self.canvas), dtype=np.float32),
        })

        self.sizes = self.cluster_areas = self.nets = None
        self.areas = self.adj = None
        self.grid = self.corners = self.next_macro = None

    def _set_problem(self, prob):
        self.sizes, self.cluster_areas, self.nets = prob
        self.areas = ([float(w * h) for (w, h) in self.sizes]
                      + [float(a) for a in self.cluster_areas])
        A = np.zeros((self.N, self.N), dtype=np.float32)
        for net in self.nets:
            for a in net:
                for b in net:
                    if a != b:
                        A[a, b] = 1.0
        self.adj = A

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
        X = np.zeros((self.N, NODE_FEATS), dtype=np.float32)
        for i in range(self.M):
            X[i, 0] = 1.0                       # is_macro
            w, h = self.sizes[i]
            X[i, 4] = w / problem.SIZE_MAX
            X[i, 5] = h / problem.SIZE_MAX
            if self.corners[i] is not None:
                X[i, 2] = 1.0                   # is_placed
                r, c = self.corners[i]
                X[i, 6] = (c + w / 2.0) / self.canvas
                X[i, 7] = (r + h / 2.0) / self.canvas
            if i == self.next_macro:
                X[i, 3] = 1.0                   # is_current
        for i in range(self.M, self.N):
            X[i, 1] = 1.0                       # is_cluster
            X[i, 4] = self.cluster_areas[i - self.M] / problem.AREA_MAX
        return {"nodes": X, "adj": self.adj.copy(),
                "grid": self.grid.copy().astype(np.float32)}

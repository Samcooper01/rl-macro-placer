"""V1 macro-placement environment (Gymnasium).

Place NUM_MACROS point macros one at a time on a CANVAS x CANVAS grid.
The agent chooses a grid cell for the next macro. When all are placed, the
terminal reward is the negative HPWL (Half-Perimeter WireLength) of the netlist.
A per-step penalty discourages placing on an occupied cell (overlap is allowed
but punished in V1; action masking is a v2 upgrade).

Observation (flat vector), designed so the policy can actually learn where to
place the next macro:
  - occupancy grid, flattened                     -> canvas*canvas
  - neighbor-position grid, flattened             -> canvas*canvas
      (cells where the CURRENT macro's already-placed neighbors sit; this is
       what tells the agent where to place to keep its connections short)
  - one-hot of which macro is next                -> num_macros
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import problem


class PlacementEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    OVERLAP_PENALTY = 5.0   # subtracted when a macro is placed on an occupied cell

    def __init__(self):
        super().__init__()
        self.canvas = problem.CANVAS
        self.num_macros = problem.NUM_MACROS
        self.nets = problem.NETS
        self.ncells = self.canvas * self.canvas

        # Action: which of the canvas*canvas cells to place the next macro in.
        self.action_space = spaces.Discrete(self.ncells)

        # Observation: [ occupancy | neighbor-positions | one-hot(next macro) ]
        obs_len = self.ncells + self.ncells + self.num_macros
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_len,), dtype=np.float32
        )

        # For each macro, the set of macros it shares a net with.
        self.neighbors = [set() for _ in range(self.num_macros)]
        for net in self.nets:
            for a in net:
                for b in net:
                    if a != b:
                        self.neighbors[a].add(b)

        # State (set in reset):
        self.grid = None
        self.positions = None
        self.next_macro = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((self.canvas, self.canvas), dtype=np.float32)
        self.positions = [None] * self.num_macros
        self.next_macro = 0
        return self._get_obs(), {}

    def step(self, action):
        row, col = divmod(int(action), self.canvas)
        reward = 0.0

        # Overlap is allowed but penalized (no action masking in V1).
        if self.grid[row, col] > 0:
            reward -= self.OVERLAP_PENALTY

        # Place the current macro.
        self.positions[self.next_macro] = (row, col)
        self.grid[row, col] = 1.0
        self.next_macro += 1

        terminated = self.next_macro >= self.num_macros
        if terminated:
            # The whole point: reward the finished placement by -HPWL.
            reward -= self._compute_hpwl()

        return self._get_obs(), reward, terminated, False, {}

    def action_masks(self):
        """Legal actions = free cells. Occupied cells are forbidden, so the
        agent can never place a macro on top of another."""
        return (self.grid.flatten() == 0.0)

    def _compute_hpwl(self):
        total = 0.0
        for net in self.nets:
            rows = [self.positions[m][0] for m in net if self.positions[m] is not None]
            cols = [self.positions[m][1] for m in net if self.positions[m] is not None]
            if len(rows) < 2:
                continue
            total += (max(rows) - min(rows)) + (max(cols) - min(cols))
        return total

    def _get_obs(self):
        occupancy = self.grid.flatten()

        neigh_grid = np.zeros((self.canvas, self.canvas), dtype=np.float32)
        onehot = np.zeros(self.num_macros, dtype=np.float32)
        if self.next_macro < self.num_macros:
            onehot[self.next_macro] = 1.0
            for j in self.neighbors[self.next_macro]:
                if self.positions[j] is not None:
                    r, c = self.positions[j]
                    neigh_grid[r, c] = 1.0

        return np.concatenate(
            [occupancy, neigh_grid.flatten(), onehot]
        ).astype(np.float32)

    def render(self, save_path=None, title=None):
        import matplotlib
        if save_path is not None:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5))
        for net in self.nets:
            pts = [self.positions[m] for m in net if self.positions[m] is not None]
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    (r1, c1), (r2, c2) = pts[i], pts[j]
                    ax.plot([c1, c2], [r1, r2], "b-", alpha=0.3, lw=0.8)
        for idx, p in enumerate(self.positions):
            if p is not None:
                r, c = p
                ax.plot(c, r, "rs", ms=14)
                ax.text(c, r, str(idx), ha="center", va="center",
                        color="white", fontsize=8)
        ax.set_xlim(-1, self.canvas)
        ax.set_ylim(-1, self.canvas)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title(title or f"HPWL = {self._compute_hpwl():.0f}")
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=120)
            plt.close(fig)
        else:
            plt.show()

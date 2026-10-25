"""A GNN (Graph Neural Network) feature extractor for the policy.

Runs two graph-convolution layers over the netlist (message passing with a
symmetrically-normalized adjacency), then combines the current macro's node
embedding, a global graph embedding, and features of the occupancy grid. This
gives the policy a representation of the netlist STRUCTURE that a flat
observation vector could not provide.
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, gnn_dim=32, features_dim=128):
        super().__init__(observation_space, features_dim)
        f = observation_space["nodes"].shape[1]
        gh, gw = observation_space["grid"].shape
        self.w1 = nn.Linear(f, gnn_dim)
        self.w2 = nn.Linear(gnn_dim, gnn_dim)
        self.grid_fc = nn.Linear(gh * gw, 64)
        self.out = nn.Linear(gnn_dim * 2 + 64, features_dim)
        self.act = nn.ReLU()

    @staticmethod
    def _norm_adj(adj):
        B, N, _ = adj.shape
        eye = torch.eye(N, device=adj.device).unsqueeze(0)
        A = adj + eye                                   # add self-loops
        deg = A.sum(-1).clamp(min=1.0)
        dinv = torch.diag_embed(deg.pow(-0.5))          # (B, N, N)
        return dinv @ A @ dinv

    def forward(self, obs):
        X = obs["nodes"]                                # (B, N, F)
        A = self._norm_adj(obs["adj"])                  # (B, N, N)
        H = self.act(A @ self.w1(X))
        H = self.act(A @ self.w2(H))                    # (B, N, D) node embeddings
        cur = X[:, :, 3:4]                              # is_current flag (B, N, 1)
        h_cur = (H * cur).sum(1)                        # current macro embedding
        h_glob = H.mean(1)                              # global graph embedding
        g = self.act(self.grid_fc(obs["grid"].flatten(1)))
        return self.act(self.out(torch.cat([h_cur, h_glob, g], dim=1)))

"""V5: a spatial (deconvolution) action head, as in the Google paper.

Instead of producing the 625 grid-cell logits with a plain linear layer (V4),
the policy upsamples the GNN (Graph Neural Network) latent into a 25 by 25 grid
of placement logits with a small spatial decoder (upsample + convolutions). This
lets the policy reason spatially, "place the current macro HERE", informed by the
netlist structure, which is what the plain head could not do.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from gnn_extractor import GNNExtractor


class SpatialActionHead(nn.Module):
    """Map the policy latent vector to a (canvas x canvas) grid of logits."""

    def __init__(self, latent_dim, canvas=25, ch=32):
        super().__init__()
        self.canvas = canvas
        self.ch = ch
        self.fc = nn.Linear(latent_dim, ch * 5 * 5)     # seed a 5x5 feature map
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, 1, 1)                # 1 logit per cell
        self.act = nn.ReLU()

    def forward(self, latent):
        x = self.act(self.fc(latent)).view(-1, self.ch, 5, 5)
        x = F.interpolate(x, size=(self.canvas, self.canvas), mode="bilinear",
                          align_corners=False)          # upsample 5x5 -> 25x25
        x = self.act(self.conv1(x))
        x = self.conv2(x)                               # (B, 1, 25, 25)
        return x.flatten(1)                             # (B, 625) action logits


class SpatialMaskablePolicy(MaskableActorCriticPolicy):
    """MaskablePPO policy with the GNN extractor and the spatial action head."""

    def __init__(self, *args, **kwargs):
        kwargs["features_extractor_class"] = GNNExtractor
        super().__init__(*args, **kwargs)
        # replace the default linear action head with the spatial decoder
        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.action_net = SpatialActionHead(latent_dim_pi, canvas=25)

import torch, torch.nn as nn, torch.nn.functional as F
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from gnn_extractor import GNNExtractor


class SpatialActionHead(nn.Module):
    def __init__(self, latent_dim, canvas, ch=32):
        super().__init__()
        self.canvas, self.ch = canvas, ch
        self.fc = nn.Linear(latent_dim, ch*5*5)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, 1, 1)
        self.act = nn.ReLU()

    def forward(self, latent):
        x = self.act(self.fc(latent)).view(-1, self.ch, 5, 5)
        x = F.interpolate(x, size=(self.canvas, self.canvas), mode="bilinear",
                          align_corners=False)
        x = self.act(self.conv1(x))
        return self.conv2(x).flatten(1)


class SpatialMaskablePolicy(MaskableActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        kwargs["features_extractor_class"] = GNNExtractor
        super().__init__(*args, **kwargs)
        canvas = self.observation_space["grid"].shape[0]
        self.action_net = SpatialActionHead(self.mlp_extractor.latent_dim_pi, canvas)

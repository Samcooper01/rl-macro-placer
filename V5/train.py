"""Train MaskablePPO with the GNN extractor + spatial deconvolution action head
(V5), using the GPU for the (now larger) policy network."""

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from placement_env import PlacementEnv
from spatial_policy import SpatialMaskablePolicy


def mask_fn(env):
    return env.action_masks()


def make_env():
    return ActionMasker(PlacementEnv(gen_seed=0), mask_fn)


def make_model(env, device="cuda"):
    return MaskablePPO(SpatialMaskablePolicy, env, verbose=0, device=device)


def main():
    model = make_model(make_env())
    model.learn(total_timesteps=400_000)
    model.save("ppo_placer_v5")


if __name__ == "__main__":
    main()

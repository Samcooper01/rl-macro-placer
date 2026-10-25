"""Train MaskablePPO on RANDOM problems (V3 generalization)."""

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from placement_env import PlacementEnv


def mask_fn(env):
    return env.action_masks()


def make_env():
    return ActionMasker(PlacementEnv(gen_seed=0), mask_fn)


def main():
    model = MaskablePPO("MlpPolicy", make_env(), verbose=1, device="cpu")
    model.learn(total_timesteps=300_000)
    model.save("ppo_placer_v3")


if __name__ == "__main__":
    main()

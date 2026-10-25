"""Train MaskablePPO on the V2 problem (3-term proxy reward)."""

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from placement_env import PlacementEnv


def mask_fn(env):
    return env.action_masks()


def make_env():
    return ActionMasker(PlacementEnv(), mask_fn)


def main():
    model = MaskablePPO("MlpPolicy", make_env(), verbose=1, device="cpu")
    model.learn(total_timesteps=250_000)
    model.save("ppo_hero")


if __name__ == "__main__":
    main()

"""Train MaskablePPO with a GNN feature extractor on random problems (V4).

Uses the GPU for the policy network (device="cuda"); the force-directed
environment stays on the CPU. Note: SubprocVecEnv parallelism would speed this
up but is not available in every environment, so this is the single-env version.
"""

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from placement_env import PlacementEnv
from gnn_extractor import GNNExtractor


def mask_fn(env):
    return env.action_masks()


def make_env():
    return ActionMasker(PlacementEnv(gen_seed=0), mask_fn)


def make_model(env, device="cuda"):
    policy_kwargs = dict(features_extractor_class=GNNExtractor)
    return MaskablePPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs,
                       verbose=0, device=device)


def main():
    model = make_model(make_env())
    model.learn(total_timesteps=300_000)
    model.save("ppo_placer_v4")


if __name__ == "__main__":
    main()

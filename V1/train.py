"""Train a MaskablePPO agent on the fixed V1 placement problem.

Action masking forbids placing on an occupied cell, so overlap is impossible
(the agent only ever chooses among free cells). This is the approach used in
the Google graph-placement paper.
"""

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from placement_env import PlacementEnv


def mask_fn(env):
    return env.action_masks()


def make_env():
    return ActionMasker(PlacementEnv(), mask_fn)


def evaluate(model, episodes=20):
    """Average HPWL of the (deterministic, masked) learned policy."""
    env = PlacementEnv()
    hpwls = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(
                obs, action_masks=env.action_masks(), deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
        hpwls.append(env._compute_hpwl())
    return sum(hpwls) / len(hpwls)


def main():
    model = MaskablePPO("MlpPolicy", make_env(), verbose=1, device="cpu")
    model.learn(total_timesteps=300_000)
    model.save("ppo_placer")
    print(f"\nlearned-policy mean HPWL: {evaluate(model):.1f}")
    print("(random agent HPWL is ~100; a compact clustered layout is ~15)")


if __name__ == "__main__":
    main()

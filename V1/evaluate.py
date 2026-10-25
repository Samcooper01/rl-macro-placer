"""Load the trained agent, run one masked placement, print HPWL, and render."""

from sb3_contrib import MaskablePPO
from placement_env import PlacementEnv


def main():
    env = PlacementEnv()
    model = MaskablePPO.load("ppo_placer")
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(
            obs, action_masks=env.action_masks(), deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated
    print(f"HPWL: {env._compute_hpwl():.0f}")
    env.render()


if __name__ == "__main__":
    main()

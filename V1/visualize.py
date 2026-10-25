"""Train briefly and save before/after placement images (no display needed)."""

from sb3_contrib import MaskablePPO
from train import make_env
from placement_env import PlacementEnv


def rollout_random(env):
    obs, _ = env.reset()
    done = False
    while not done:
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        done = terminated or truncated


def rollout_policy(env, model):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(
            obs, action_masks=env.action_masks(), deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated


def main():
    env = PlacementEnv()
    rollout_random(env)
    env.render(save_path="placement_random.png",
               title=f"Random placement  (HPWL = {env._compute_hpwl():.0f})")

    model = MaskablePPO("MlpPolicy", make_env(), verbose=0, device="cpu")
    model.learn(total_timesteps=200_000)
    model.save("ppo_placer")

    rollout_policy(env, model)
    env.render(save_path="placement_learned.png",
               title=f"Learned placement  (HPWL = {env._compute_hpwl():.0f})")
    print("saved placement_random.png and placement_learned.png")


if __name__ == "__main__":
    main()

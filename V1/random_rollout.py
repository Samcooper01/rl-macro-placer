"""Validation harness: run the environment with a RANDOM agent.

Run this BEFORE writing any RL. It checks that the environment works:
episodes complete, rewards are finite and sensibly ranged, and (the key
check) that a good hand-placement scores higher than a bad one. If that
ranking is wrong, the environment is lying and no agent can learn from it.
"""

import numpy as np
from placement_env import PlacementEnv


def run_random(episodes=200):
    env = PlacementEnv()
    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        rewards.append(total)
    rewards = np.array(rewards)
    print(f"random agent over {episodes} episodes:")
    print(f"  reward  mean {rewards.mean():.2f}  min {rewards.min():.2f}  max {rewards.max():.2f}")
    return env


if __name__ == "__main__":
    env = run_random()
    # After this runs cleanly, add a good-vs-bad sanity check:
    # manually set env.positions to a compact clustered layout and to a
    # scattered one, call env._compute_hpwl() on each, and confirm the
    # clustered layout has the LOWER HPWL. Then you are ready for PPO.

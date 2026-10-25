"""Train and save full-chip placement images (macros + clusters + metrics)."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from sb3_contrib import MaskablePPO
from train import make_env
from placement_env import PlacementEnv


def render_full(env, save_path, title):
    pos = env.full_placement(seed=0)
    M = env.M
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for net in env.nets:
        p = pos[np.asarray(net)]
        cen = p.mean(axis=0)
        for q in p:
            ax.plot([cen[0], q[0]], [cen[1], q[1]], "b-", alpha=0.12, lw=0.6)
    for i in range(M):
        r, c = env.corners[i]
        w, h = env.sizes[i]
        ax.add_patch(Rectangle((c, r), w, h, color="red", alpha=0.85, zorder=2))
    ax.scatter(pos[M:, 0], pos[M:, 1], s=70, c="green", marker="o", zorder=3)
    ax.set_xlim(0, env.canvas)
    ax.set_ylim(0, env.canvas)
    ax.set_aspect("equal")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=115)
    plt.close(fig)


def run(env, model=None):
    obs, _ = env.reset()
    done = False
    while not done:
        if model is not None:
            a, _ = model.predict(obs, action_masks=env.action_masks(),
                                 deterministic=True)
        else:
            a = np.random.choice(np.flatnonzero(env.action_masks()))
        obs, _, terminated, truncated, _ = env.step(int(a))
        done = terminated or truncated


def main():
    env = PlacementEnv()

    run(env)
    wl, cg, dn = env.cost_terms()
    render_full(env, "placement_random.png",
                f"Random placement   (WL {wl:.0f}, cong {cg:.2f}, dens {dn:.1f})")

    model = MaskablePPO("MlpPolicy", make_env(), verbose=0, device="cpu")
    model.learn(total_timesteps=250_000)
    model.save("ppo_placer_v2")

    run(env, model)
    wl, cg, dn = env.cost_terms()
    render_full(env, "placement_learned.png",
                f"Learned placement   (WL {wl:.0f}, cong {cg:.2f}, dens {dn:.1f})")
    print(f"learned terms: WL {wl:.0f}  cong {cg:.2f}  dens {dn:.1f}")
    print("saved placement_random.png and placement_learned.png")


if __name__ == "__main__":
    main()

"""Train on random problems, measure generalization on a held-out test set, and
render the learned policy on unseen problems."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from sb3_contrib import MaskablePPO
from train import make_env
from placement_env import PlacementEnv
import problem


def make_test_set(n=30, seed=999):
    rng = np.random.default_rng(seed)      # a DIFFERENT stream from training
    return [problem.generate_problem(rng) for _ in range(n)]


def run(env, prob, model=None):
    obs, _ = env.reset(options={"problem": prob})
    done = False
    while not done:
        if model is not None:
            a, _ = model.predict(obs, action_masks=env.action_masks(),
                                 deterministic=True)
        else:
            a = np.random.choice(np.flatnonzero(env.action_masks()))
        obs, _, terminated, truncated, _ = env.step(int(a))
        done = terminated or truncated
    return env._full_cost()


def evaluate(model, problems):
    env = PlacementEnv()
    learned = np.array([run(env, p, model) for p in problems])
    rand = np.array([run(env, p, None) for p in problems])
    return learned.mean(), rand.mean()


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
    ax.set_xlim(0, env.canvas); ax.set_ylim(0, env.canvas); ax.set_aspect("equal")
    ax.set_title(title)
    plt.tight_layout(); plt.savefig(save_path, dpi=115); plt.close(fig)


def main():
    test = make_test_set()

    model = MaskablePPO("MlpPolicy", make_env(), verbose=0, device="cpu")
    model.learn(total_timesteps=300_000)
    model.save("ppo_placer_v3")

    lm, rm = evaluate(model, test)
    print(f"HELD-OUT (unseen) problems, n={len(test)}:")
    print(f"  learned mean cost {lm:.1f}   random mean cost {rm:.1f}   "
          f"improvement {100*(rm-lm)/rm:.0f}%")

    env = PlacementEnv()
    for i in (0, 1):
        run(env, test[i], model)
        wl, cg, dn = env.cost_terms()
        render_full(env, f"heldout_{i}.png",
                    f"Held-out problem {i} (never trained on)\nWL {wl:.0f}  cong {cg:.2f}  dens {dn:.1f}")
    print("saved heldout_0.png and heldout_1.png")


if __name__ == "__main__":
    main()

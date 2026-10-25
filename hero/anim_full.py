"""Full-method animation: the RL agent places sized macros one at a time, then a
force-directed pass settles the standard-cell clusters around them, with the
three-term proxy metrics (wirelength, congestion, density) shown throughout.
Saves demo.gif."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from placement_env import PlacementEnv
import problem, metrics
import force_directed as fd

CANVAS = problem.CANVAS
M = problem.NUM_MACROS


def get_model(env):
    if os.path.exists("ppo_hero.zip"):
        return MaskablePPO.load("ppo_hero")
    m = MaskablePPO("MlpPolicy", ActionMasker(PlacementEnv(), lambda e: e.action_masks()),
                    verbose=0, device="cpu")
    m.learn(total_timesteps=150_000)
    m.save("ppo_hero")
    return m


def fd_snapshots(macro_pos, macro_sizes, num_macros, num_clusters, nets, canvas,
                 iters=fd.ITERS, snap_every=4, seed=0):
    """Force-directed placement that records intermediate cluster positions
    (same short-range model as force_directed.py)."""
    rng = np.random.default_rng(seed)
    N = num_macros + num_clusters
    pos = np.zeros((N, 2))
    pos[:num_macros] = np.asarray(macro_pos, float)
    center = pos[:num_macros].mean(axis=0)
    pos[num_macros:] = center + rng.standard_normal((num_clusters, 2)) * 1.5
    mv = slice(num_macros, N)
    snaps = [pos.copy()]
    for it in range(iters):
        force = np.zeros((N, 2))
        for net in nets:
            idx = np.asarray(net)
            force[idx] += fd.K_ATT * (pos[idx].mean(0) - pos[idx])
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1)) + 1e-6
        close = (dist < fd.R_REP) & (dist > 1e-3)
        push = np.where(close[:, :, None],
                        fd.K_REP * ((fd.R_REP - dist) / fd.R_REP)[:, :, None] * diff / dist[:, :, None],
                        0.0)
        force += push.sum(1)
        for m in range(num_macros):
            mx, my = pos[m]; w, h = macro_sizes[m]
            hx, hy = w / 2 + fd.GAP, h / 2 + fd.GAP
            for k in range(num_macros, N):
                dx, dy = pos[k, 0] - mx, pos[k, 1] - my
                if abs(dx) < hx and abs(dy) < hy:
                    px, py = hx - abs(dx), hy - abs(dy)
                    if px < py:
                        force[k, 0] += fd.K_BLOCK * (1 if dx >= 0 else -1) * px
                    else:
                        force[k, 1] += fd.K_BLOCK * (1 if dy >= 0 else -1) * py
        pos[mv] += fd.STEP * force[mv]
        pos[mv] = np.clip(pos[mv], 0, canvas)
        if (it + 1) % snap_every == 0:
            snaps.append(pos.copy())
    # iterated legalization to a fixed point (matches force_directed_place)
    for _ in range(40):
        moved = False
        for m in range(num_macros):
            mx, my = pos[m]; w, h = macro_sizes[m]
            hx, hy = w / 2 + fd.GAP, h / 2 + fd.GAP
            for k in range(num_macros, N):
                dx, dy = pos[k, 0] - mx, pos[k, 1] - my
                if abs(dx) < hx and abs(dy) < hy:
                    if hx - abs(dx) < hy - abs(dy):
                        pos[k, 0] = mx + (hx if dx >= 0 else -hx)
                    else:
                        pos[k, 1] = my + (hy if dy >= 0 else -hy)
                    moved = True
        pos[num_macros:] = np.clip(pos[num_macros:], 0, canvas)
        if not moved:
            break
    snaps.append(pos.copy())
    return snaps


def main():
    env = PlacementEnv()
    model = get_model(env)

    # run the learned policy, recording macro corners after each placement
    obs, _ = env.reset()
    macro_stages = []
    done = False
    while not done:
        a, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        obs, _, term, trunc, _ = env.step(int(a))
        macro_stages.append([env.corners[i] for i in range(M)])
        done = term or trunc

    centers = env._macro_centers()
    snaps = fd_snapshots(centers, env.sizes, M, env.C, env.nets, CANVAS)
    areas = env.areas

    # build the frame list: (macro_corners, cluster_positions_or_None, phase_label)
    frames = []
    for stage in macro_stages:
        frames.append((stage, None, "placing macros"))
    for s in snaps:
        frames.append(([env.corners[i] for i in range(M)], s, "placing standard cells"))
    frames += [frames[-1]] * 4   # hold the final frame

    fig, ax = plt.subplots(figsize=(5.6, 5.6))

    def draw(i):
        corners, clpos, phase = frames[i]
        ax.clear()
        full = clpos
        if full is not None:
            for net in env.nets:
                p = full[np.asarray(net)]; cen = p.mean(0)
                for q in p:
                    ax.plot([cen[0], q[0]], [cen[1], q[1]], "b-", alpha=0.12, lw=0.6)
            ax.scatter(full[M:, 0], full[M:, 1], s=60, c="green", marker="o", zorder=3)
        for j in range(M):
            if corners[j] is not None:
                r, c = corners[j]; w, h = env.sizes[j]
                ax.add_patch(Rectangle((c, r), w, h, color="red", alpha=0.85, zorder=2))
        ax.set_xlim(0, CANVAS); ax.set_ylim(0, CANVAS); ax.set_aspect("equal")
        if full is not None:
            wl = metrics.hpwl(full, env.nets)
            cg = metrics.congestion(full, env.nets, CANVAS)
            dn = metrics.density(full, areas, CANVAS)
            ax.set_title(f"{phase}\nwirelength {wl:.0f}   congestion {cg:.2f}   density {dn:.1f}")
        else:
            ax.set_title(f"{phase}\n(macros placed one at a time by the RL agent)")

    ani = animation.FuncAnimation(fig, draw, frames=len(frames), interval=500)
    ani.save("demo.gif", writer="pillow", fps=2.2)
    plt.close(fig)
    print(f"saved demo.gif ({len(frames)} frames)")


if __name__ == "__main__":
    main()

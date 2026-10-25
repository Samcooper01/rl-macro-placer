"""Hero animation for the write-up (docs/assets/hero.gif and hero-dark.gif).

The trained V2 agent places its four macros one at a time, then the standard-cell clusters
settle around them. Unlike force_directed.py, clusters are kept clear of the macros after every
step (not only at the end), so no frame shows a cluster on top of a macro. The final scores
therefore differ slightly from V2's reported result. Drawn once per page theme.

Run from this folder: python3 animate_hero.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from PIL import Image
from sb3_contrib import MaskablePPO

from placement_env import PlacementEnv
import metrics
import force_directed as fd

THEMES = {  # colours from docs/style.css
    "light": dict(bg="#fffbf3", text="#1a1612", muted="#5e5446", grid="#ddd0b8", macro="#b91c1c", cluster="#3f6212"),
    "dark":  dict(bg="#1f1a13", text="#f1e8d8", muted="#b0a28b", grid="#3a3125", macro="#f87171", cluster="#a3e635"),
}
SNAP_EVERY = 3


def keep_out(pos, num_macros, macro_sizes):
    """Move any cluster inside a macro's keep-out box to the nearest edge (one pass)."""
    for m in range(num_macros):
        mx, my = pos[m]
        hx, hy = macro_sizes[m][0] / 2.0 + fd.GAP, macro_sizes[m][1] / 2.0 + fd.GAP
        d = pos[num_macros:] - pos[m]
        inside = (np.abs(d[:, 0]) < hx) & (np.abs(d[:, 1]) < hy)
        for i in np.flatnonzero(inside):
            k = num_macros + i
            dx, dy = d[i]
            if hx - abs(dx) < hy - abs(dy):
                pos[k, 0] = mx + (hx if dx >= 0 else -hx)
            else:
                pos[k, 1] = my + (hy if dy >= 0 else -hy)


def settle(macro_pos, macro_sizes, num_macros, num_clusters, nets, canvas, seed=0):
    """V2's force-directed rule with the keep-out enforced after every step; returns snapshots."""
    rng = np.random.default_rng(seed)
    N = num_macros + num_clusters
    pos = np.zeros((N, 2))
    pos[:num_macros] = macro_pos
    pos[num_macros:] = pos[:num_macros].mean(axis=0) + rng.standard_normal((num_clusters, 2)) * 1.5
    keep_out(pos, num_macros, macro_sizes)
    snaps = [pos.copy()]
    for it in range(fd.ITERS):
        force = np.zeros((N, 2))
        for net in nets:
            idx = np.asarray(net)
            force[idx] += fd.K_ATT * (pos[idx].mean(axis=0) - pos[idx])
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1)) + 1e-6
        close = (dist < fd.R_REP) & (dist > 1e-3)
        force += np.where(close[:, :, None],
                          fd.K_REP * ((fd.R_REP - dist) / fd.R_REP)[:, :, None] * diff / dist[:, :, None],
                          0.0).sum(axis=1)
        pos[num_macros:] += fd.STEP * force[num_macros:]
        pos[num_macros:] = np.clip(pos[num_macros:], 0, canvas)
        for _ in range(5):
            keep_out(pos, num_macros, macro_sizes)
        if (it + 1) % SNAP_EVERY == 0:
            snaps.append(pos.copy())
    return snaps


def draw(ax, env, corners, pos, c):
    ax.clear()
    ax.set_facecolor(c["bg"])
    if pos is not None:
        for net in env.nets:
            p = pos[np.asarray(net)]
            cen = p.mean(0)
            for q in p:
                ax.plot([cen[0], q[0]], [cen[1], q[1]], color=c["muted"], alpha=0.35, lw=0.7, zorder=1)
        for k in range(env.M, env.M + env.C):
            ax.add_patch(Circle(pos[k], 0.55, color=c["cluster"], zorder=3))
    for j in range(env.M):
        if corners[j] is not None:
            r, col = corners[j]
            w, h = env.sizes[j]
            ax.add_patch(Rectangle((col, r), w, h, color=c["macro"], alpha=0.9, zorder=2))
    ax.set_xlim(0, env.canvas); ax.set_ylim(0, env.canvas); ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_color(c["grid"])
    ax.tick_params(colors=c["muted"], labelsize=9)
    if pos is None:
        placed = sum(x is not None for x in corners)
        title = f"placing macros: {placed} of {env.M}\n(placed one at a time by the trained agent)"
    else:
        title = (f"placing standard cells\nwirelength {metrics.hpwl(pos, env.nets):.0f}   "
                 f"congestion {metrics.congestion(pos, env.nets, env.canvas):.2f}   "
                 f"density {metrics.density(pos, env.areas, env.canvas):.1f}")
    ax.set_title(title, color=c["text"], fontsize=11)


def main():
    plt.rcParams["font.family"] = "serif"
    env = PlacementEnv()
    model = MaskablePPO.load("ppo_placer_v2", device="cpu")
    obs, _ = env.reset()
    stages, done = [], False
    while not done:
        a, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        obs, _, done, _, _ = env.step(int(a))
        stages.append(list(env.corners))
    snaps = settle(env._macro_centers(), env.sizes, env.M, env.C, env.nets, env.canvas)
    final = snaps[-1]
    print(f"final: wirelength {metrics.hpwl(final, env.nets):.0f}  "
          f"congestion {metrics.congestion(final, env.nets, env.canvas):.2f}  "
          f"density {metrics.density(final, env.areas, env.canvas):.1f}")
    frames = [(s, None) for s in stages] + [(stages[-1], p) for p in snaps]
    frames += [frames[-1]] * 6                       # hold the finished layout

    for name, c in THEMES.items():
        fig, ax = plt.subplots(figsize=(5.6, 5.6), dpi=100)
        fig.patch.set_facecolor(c["bg"])
        images = []
        for corners, pos in frames:
            draw(ax, env, corners, pos, c)
            fig.canvas.draw()
            rgb = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
            images.append(rgb.quantize(colors=96, method=0, dither=0))
        plt.close(fig)
        out = "../docs/assets/hero.gif" if name == "light" else "../docs/assets/hero-dark.gif"
        images[0].save(out, save_all=True, append_images=images[1:], duration=450, loop=0, optimize=True)
        print(f"saved {out} ({len(images)} frames)")


if __name__ == "__main__":
    main()

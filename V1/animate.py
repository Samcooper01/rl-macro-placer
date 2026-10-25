"""Watch the trained agent place macros one at a time -> placement.gif"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from sb3_contrib import MaskablePPO
from placement_env import PlacementEnv


def main():
    env = PlacementEnv()
    model = MaskablePPO.load("ppo_placer")

    obs, _ = env.reset()
    frames = [list(env.positions)]          # snapshot after each placement
    done = False
    while not done:
        action, _ = model.predict(
            obs, action_masks=env.action_masks(), deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        frames.append(list(env.positions))
        done = terminated or truncated

    fig, ax = plt.subplots(figsize=(5, 5))

    def draw(i):
        positions = frames[i]
        ax.clear()
        for net in env.nets:
            pts = [positions[m] for m in net if positions[m] is not None]
            for a in range(len(pts)):
                for b in range(a + 1, len(pts)):
                    (r1, c1), (r2, c2) = pts[a], pts[b]
                    ax.plot([c1, c2], [r1, r2], "b-", alpha=0.3, lw=0.8)
        for idx, p in enumerate(positions):
            if p is not None:
                r, c = p
                ax.plot(c, r, "rs", ms=14)
                ax.text(c, r, str(idx), ha="center", va="center",
                        color="white", fontsize=8)
        placed = sum(1 for p in positions if p is not None)
        ax.set_xlim(-1, env.canvas)
        ax.set_ylim(-1, env.canvas)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        title = f"step {i}: {placed}/{env.num_macros} macros placed"
        if placed == env.num_macros:
            title += f"   (HPWL = {env._compute_hpwl():.0f})"
        ax.set_title(title)

    ani = animation.FuncAnimation(fig, draw, frames=len(frames), interval=700)
    ani.save("placement.gif", writer="pillow", fps=1.5)
    plt.close(fig)
    print("saved placement.gif")


if __name__ == "__main__":
    main()

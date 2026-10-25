"""Train the hero model once, then render the static placement image and the
animated demo.gif. Kept separate from visualize.py so we never retrain by
accident when only re-rendering."""

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from placement_env import PlacementEnv
import visualize as V

STEPS = 300_000

env = PlacementEnv()

# 1) train the macro-placement policy on the hero problem
model = MaskablePPO("MlpPolicy",
                    ActionMasker(PlacementEnv(), lambda e: e.action_masks()),
                    verbose=0, device="cpu", n_steps=1024)
model.learn(total_timesteps=STEPS)
model.save("ppo_hero")
print("trained ppo_hero")

# 2) static before/after images from this model
V.run(env)
wl, cg, dn = env.cost_terms()
V.render_full(env, "placement_random.png",
              f"Random placement   (WL {wl:.0f}, cong {cg:.2f}, dens {dn:.1f})")
V.run(env, model)
wl, cg, dn = env.cost_terms()
V.render_full(env, "placement_learned.png",
              f"Learned placement   (WL {wl:.0f}, cong {cg:.2f}, dens {dn:.1f})")
print(f"learned: WL {wl:.0f}  cong {cg:.2f}  dens {dn:.1f}")

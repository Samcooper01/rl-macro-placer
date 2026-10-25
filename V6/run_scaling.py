"""Scaling study: does the GNN advantage emerge as problem size grows?

For each scale (number of macros / clusters / canvas), train a plain baseline
policy and the GNN + deconvolution policy, and measure held-out improvement over
random. If the GNN's relative advantage grows with scale, that reproduces the
paper's finding (their designs are large); if not, scale is not the missing piece
at this training budget.
"""
import json
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from placement_env import PlacementEnv
from spatial_policy import SpatialMaskablePolicy
import problem_gen

SCALES = [(4, 12, 25), (6, 18, 31), (8, 24, 35), (10, 30, 40), (12, 36, 43), (14, 42, 47)]
STEPS = 100_000
N_TEST = 40


def mask_fn(e):
    return e.action_masks()


def make_env(m, c, cv, seed=0):
    return ActionMasker(PlacementEnv(m, c, cv, gen_seed=seed), mask_fn)


def test_set(m, c, cv, n=N_TEST, seed=999):
    rng = np.random.default_rng(seed)
    return [problem_gen.generate_problem(rng, m, c, cv) for _ in range(n)]


def run(env, prob, model=None):
    obs, _ = env.reset(options={"problem": prob})
    done = False
    while not done:
        if model is not None:
            a, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        else:
            a = np.random.choice(np.flatnonzero(env.action_masks()))
        obs, _, t, tr, _ = env.step(int(a))
        done = t or tr
    return env._full_cost()


def evaluate(model, m, c, cv, problems):
    env = PlacementEnv(m, c, cv)
    learned = np.array([run(env, p, model) for p in problems])
    rand = np.array([run(env, p, None) for p in problems])
    return learned.mean(), rand.mean()


def main():
    results = {}
    for (m, c, cv) in SCALES:
        problems = test_set(m, c, cv)
        row = {}
        for name, policy in [("baseline", "MultiInputPolicy"),
                             ("gnn_deconv", SpatialMaskablePolicy)]:
            model = MaskablePPO(policy, make_env(m, c, cv), device="cuda", verbose=0)
            model.learn(STEPS)
            lm, rm = evaluate(model, m, c, cv, problems)
            imp = 100 * (rm - lm) / rm
            row[name] = imp
            print(f"scale M={m:2d}: {name:10s} improvement {imp:5.1f}%  "
                  f"(learned {lm:.1f} vs random {rm:.1f})", flush=True)
        results[m] = row

    print("SUMMARY:", json.dumps(results), flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ms = [s[0] for s in SCALES]
    plt.figure(figsize=(6, 4.2))
    plt.plot(ms, [results[m]["baseline"] for m in ms], "o-", label="flat baseline")
    plt.plot(ms, [results[m]["gnn_deconv"] for m in ms], "s-", label="GNN + deconvolution")
    plt.xlabel("problem scale (number of macros)")
    plt.ylabel("held-out improvement over random (%)")
    plt.title("Does the GNN advantage emerge with scale?")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("scaling_study.png", dpi=120)
    print("saved scaling_study.png", flush=True)


if __name__ == "__main__":
    main()

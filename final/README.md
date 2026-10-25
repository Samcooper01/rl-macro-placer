# final , the code the final simulation ran

This folder consolidates, in one place, every file used by the project's final
simulation: the V6 scaling study. That study is the second, six-size run across
macro counts that tested whether a Graph Neural Network (GNN) policy pulls ahead of a
flat policy as problems grow. It did not show the first run's rising trend, but run-to-run
noise is as large as the effects, so neither run settles the question. It is the last
and most complete point in the V1 to V6 arc.

Nothing here is new code. These are copies of the V6 files, gathered under a
clearer name so the final version can be read as a self-contained whole without
tracing which building blocks it reused from earlier versions. The authoritative
copies still live in [`../V6/`](../V6/); the write-up is in
[`../RESULTS.md`](../RESULTS.md) and the mechanics in
[`../HOW_IT_WORKS.md`](../HOW_IT_WORKS.md).

## The files, in reading order (bottom-up)

Read them in dependency order, so each file only uses things you have already seen:

1. **`problem_gen.py`** , generates a *random* problem each episode: the objects
   (macros then clusters), their sizes and areas, and the netlist connecting them.
   This is the V3-and-later replacement for the single static `problem.py`, and it
   is what forces the policy to generalize rather than memorize one layout.
2. **`metrics.py`** , the three reward primitives: Half-Perimeter WireLength
   (HPWL), a Rectangular Uniform wire DensitY (RUDY) congestion estimate, and a
   coarse-grid density estimate.
3. **`force_directed.py`** , the physics pass that settles the standard-cell
   clusters around the placed macros (springs, short-range repulsion, macro
   keep-outs, iterated legalization).
4. **`placement_env.py`** , the Gymnasium environment that ties the above together:
   the state (occupancy grid plus the netlist as a graph), the action (a grid
   cell), the legality mask, and the terminal three-term proxy reward.
5. **`gnn_extractor.py`** , the Graph Neural Network (GNN) encoder that reads the
   netlist as a graph (two graph-convolution layers of message passing) and
   produces the state representation the policy acts on.
6. **`spatial_policy.py`** , the spatial action head (it decodes the encoder's
   latent into one logit per grid cell) wired into a maskable actor-critic policy.
7. **`run_scaling.py`** , the entry point: the final scaling experiment itself. It
   generates train, validation, and test problems at each macro count, trains the
   policy, and measures the learned-versus-random advantage on the held-out test
   set across scales.

## Running it

```bash
# from the repository root
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cd final && python3 run_scaling.py
```

The environment step and the force-directed reward run on the Central Processing
Unit (CPU); the policy network uses a Graphics Processing Unit (GPU) if one is
available.

## The result it produced

`scaling_study.png` is the figure from this run, and `results.html` is its results
page. The headline: at accessible scale there is no reliable Graph Neural Network
(GNN) advantage over the simple baseline. The learned-versus-random gap stays
within noise and both decline toward random as the macro count grows. The earlier
three-point trend that suggested otherwise did not survive this larger,
higher-replication study.
</content>

# How this implementation works, and how it compares to the paper

This document explains the mechanics of *this* reinforcement-learning (RL) macro
placer end to end, then states precisely where it matches the method in the paper
and where it deliberately differs. The paper is Mirhoseini, Goldie, et al., "A
graph placement methodology for fast chip design" (Nature 2021; arXiv
[2004.10746](https://arxiv.org/abs/2004.10746)), with the open-source
[Circuit Training](https://github.com/google-research/circuit_training) release.

For the results and the honest negative findings, see
[RESULTS.md](RESULTS.md). This document is about *how it works*, not how well.

---

## 1. The core idea

Macro placement is reframed as a sequence of decisions instead of one large
optimization. The agent places the hard macros one at a time. After the last
macro lands, a force-directed pass drops the standard-cell clusters around them,
and the finished floorplan is scored by a cheap three-term proxy. That single
score is the reward. Reinforcement learning (RL) then adjusts the placement policy
so that decisions leading to better-scoring floorplans become more likely.

## 2. One episode, step by step

1. **Reset.** A problem appears: macro sizes, standard-cell clusters, and a
   netlist connecting them (`problem.py`).
2. **Place loop.** For each macro in turn, the agent observes the partial
   floorplan, the policy produces a preference over grid cells, illegal cells are
   masked out, one cell is chosen, and the macro is placed there
   (`placement_env.py`).
3. **Complete and score.** When all macros are placed, the force-directed pass
   settles the clusters (`force_directed.py`), and the three proxy metrics are
   computed over the whole placement (`metrics.py`).
4. **Reward and learn.** The reward is the negative weighted sum of the three
   metrics, handed back only at the end of the episode. The training algorithm
   uses it to update the policy.

## 3. The environment (state, action, legality)

- **State (observation).** The current occupancy grid, which macro is next, and
  (from V4 on) the netlist as a graph: node features plus an adjacency matrix.
- **Action.** Choose one grid cell for the next macro's placement. On the 25x25
  canvas that is one of 625 choices.
- **Action masking.** Before the policy chooses, the environment computes which
  cells are legal (the macro's footprint fits on the canvas and does not overlap
  any already-placed macro) and forbids the rest. This makes overlap-free
  placement a hard guarantee, not something the agent has to learn. Masked
  training uses MaskablePPO from sb3-contrib.
- **Macro order.** Macros are placed in a fixed netlist-index order.

## 4. The reward: a three-term proxy

A real score would need routing and timing, which are far too slow to run inside a
loop that plays hundreds of thousands of episodes. So three cheap proxies stand in
(`metrics.py`):

- **Wirelength:** Half-Perimeter WireLength (HPWL), the sum over nets of each net's
  bounding-box half-perimeter.
- **Congestion:** a Rectangular Uniform wire DensitY (RUDY) estimate on a 25x25
  grid. Each net spreads its wirelength demand over the cells of its bounding box;
  congestion is the mean of the most-demanded cells.
- **Density:** each object's area is rasterized onto a coarse 8x8 grid; density is
  the mean of the most-occupied cells, which rewards spreading area out.

The reward is `-(0.1 * wirelength + 10.0 * congestion + 2.0 * density)`, given only
at the end of the episode.

## 5. Force-directed standard-cell placement

The agent places only the macros. The standard-cell clusters are settled by a
small physics simulation that runs every episode (`force_directed.py`):

- springs pull connected objects toward the centroid of every net they share,
- a short-range repulsion pushes apart only objects closer than a fixed radius, so
  clusters nestle among the macros instead of being flung to the canvas edges,
- placed macros act as keep-out blockages the clusters must sit around,
- a final legalization pass iterates to a fixed point so no cluster ends up inside
  a macro.

## 6. The learning algorithm: Proximal Policy Optimization

Training uses Proximal Policy Optimization (PPO), an actor-critic policy-gradient
method, through the MaskablePPO variant that respects the legality mask. This
section builds it up from the basic idea, because the clipping that gives Proximal Policy
Optimization (PPO) its name only makes sense once the update underneath it is clear.

### 6.1 The policy is a probability distribution

The policy is a network that takes the state and outputs a number for every grid
cell, which a softmax turns into a probability distribution over cells. The agent
samples from that distribution rather than always taking the top choice, which is
what makes it explore. Training's whole job is to sharpen that distribution toward
good cells. Learning means adjusting the network's weights so the probabilities
change; every "update" is one small step of gradient descent on those weights.

### 6.2 The basic update: reinforce what worked

Reinforcement learning has no labelled correct action; the only feedback is the
reward. So the update is indirect. Play an episode, record for each placement the
log-probability the policy assigned to the cell it actually chose, and record the
return at the end. Then form a loss that, when back-propagated, makes good actions
more likely:

  loss = − (return) × (sum over placements of log-probability of the chosen cell)

Read it by sign. A good episode (high return) makes the network raise the
probability of every cell it chose; a bad episode lowers them. The return scales
how hard. That is the whole engine: sample your own data by acting, then do more
of what paid off.

### 6.3 The refinement: advantage instead of raw return

Weighting every placement by the same episode return is a crude, high-variance
signal, and in this project the return is always negative (it is a negative cost),
so the raw version pushes every placement's probability down by differing amounts.
The fix is to subtract a baseline: the critic (a value head on the same network)
learns to predict the expected return from a state, and the update uses the
**advantage**, the actual outcome minus that prediction. Now a placement is judged
against what is typical *from its own state*, so placements that beat expectations
go up and worse-than-typical ones go down. In the loss above, the return is
replaced by the advantage. Proximal Policy Optimization (PPO) estimates the advantage with Generalized Advantage
Estimation (GAE), which trades off bias and variance in that estimate.

### 6.3.1 A closer look at Generalized Advantage Estimation

The advantage estimate sits on a bias-variance spectrum, and Generalized Advantage
Estimation (GAE) is a tunable point on it. Start from the per-step
Temporal Difference (TD) error:

  delta at step t = reward at t + gamma × V(next state) − V(this state)

It measures how much one step beat the critic's expectation, using the critic's own
value of the next state as a one-step lookahead. Here gamma (γ) is the discount
factor. There are two extreme ways to turn these into an advantage:

- use only the single delta at step t (a one-step bootstrap): low variance, but
  biased by the critic's imperfect value estimate,
- use the full actual return from step t to the end minus V(this state): unbiased,
  but high variance, because it depends on all the randomness in the rest of the
  episode.

Generalized Advantage Estimation (GAE) blends the two with a smoothing parameter
lambda (λ) between 0 and 1:

  advantage at step t = sum over k ≥ 0 of (gamma × lambda) to the power k × (delta at step t+k)

That is a discounted, exponentially-weighted sum of the future temporal-difference
errors. Lambda equal to 0 collapses to the single-step, low-variance estimate;
lambda equal to 1 recovers the full-return, high-variance one; a typical lambda
around 0.95 keeps most of the variance reduction while accepting a little bias.

In this project the reward is terminal, so every intermediate reward is zero and
only the final step carries the proxy cost. That makes each intermediate delta
purely the critic's change of opinion between two partial floorplans,
gamma × V(floorplan after this macro) − V(floorplan before it), while the real
reward enters only at the last step. Generalized Advantage Estimation (GAE) is then
exactly the machinery that propagates that final reward backward through the
sequence, weighted by (gamma × lambda) to the power k, and folds it together with
the critic's per-state value estimates so that each individual placement gets its
own advantage. That is the concrete answer to how a placement in the middle of an
episode receives a credit signal of its own rather than just the flat final return.

### 6.4 The proximal part: the clipped ratio

Plain policy gradient has a dilemma. The batch was collected with the current
policy, so a large update moves the policy somewhere the batch no longer describes,
and the step can be destructive; but tiny updates that discard each batch after one
step are wastefully sample-inefficient. Proximal Policy Optimization (PPO) reuses each batch for several gradient
steps without letting the policy run away, using one object: the probability ratio.

  ratio = (probability the new policy gives the chosen action) ÷ (probability the
  old policy gave it)

The old policy is frozen during the update; a ratio of 1 means no change. The
objective per sample is:

  minimum of ( ratio × advantage , clipped-ratio × advantage )

where the clipped ratio is squashed into the band [1 − epsilon, 1 + epsilon], with
epsilon usually 0.2. The clip flattens the objective (zero gradient) only when the
ratio has already moved *past the band in the direction the advantage wants*, so
there is no reward for pushing a favored action's probability more than about 20
percent per update. But if an update moved an action the *wrong* way, the unclipped
term is the smaller one, the minimum selects it, and its full gradient still pulls
the action back. In short, Proximal Policy Optimization (PPO) clips optimism, not correction: it caps how far the
policy can advance in the direction the noisy advantage points, but never disables
undoing a move that went the wrong way. That one-sided barrier is what keeps each
batch a valid description of the policy across several gradient steps.

### 6.5 The full objective and the loop

The complete loss adds two more terms to the clipped surrogate above: a value loss
(squared error training the critic to predict returns, since a good critic gives
good advantages) and a small entropy bonus (a reward for keeping the distribution
spread out, which preserves exploration). The training loop is then:

1. Collect a fixed number of timesteps with the current policy, recording states,
   actions, rewards, value estimates, and action probabilities. One timestep is one
   macro placement, so a batch spans many full episodes.
2. Compute advantages and returns with Generalized Advantage Estimation (GAE).
3. Freeze the old policy. For several epochs, split the batch into minibatches and
   take a gradient step on the combined loss for each, the clip keeping every step
   inside the trust band.
4. The updated policy becomes the old policy for the next collection. Repeat until
   the timestep budget is spent.

### 6.6 In this project

Only one thing changes from stock Proximal Policy Optimization (PPO): the
MaskablePPO variant forces zero probability on illegal cells before the softmax, so
both the old and new policies only ever place probability on legal cells, the ratio
is computed over legal actions only, and no learning is wasted on placements that
could never happen. The reward is terminal (the negative proxy cost, only at the
end of the episode), so every placement in an episode shares the same return, and
the advantage is what distinguishes them per placement. Training runs a single
environment; the policy network runs on a Graphics Processing Unit (GPU) from V4
on, while the force-directed reward stays on the Central Processing Unit (CPU).

## 7. The policy network across versions

The representation of the state is the axis the versions explore:

- **V1 to V3 (flat).** A plain Multi-Layer Perceptron (MLP) over a flat observation vector. Works for a fixed problem; connectivity is
  implicit and it cannot transfer across differently-shaped netlists.
- **V4 to V6 (Graph Neural Network).** A Graph Neural Network (GNN) encoder
  (`gnn_extractor.py`) reads the netlist as a graph. It runs two
  graph-convolution layers of message passing with a symmetrically-normalized
  adjacency (each object's embedding is built from its neighbors' embeddings),
  then concatenates the current macro's embedding, a global graph embedding, and
  features of the occupancy grid. This gives the policy the netlist *structure*
  the flat observation could not, and it is size- and order-agnostic, the property
  that would let one policy place many designs.
- **V5 to V6 (spatial action head).** Instead of a flat list of logits, a spatial
  head (`spatial_policy.py`) decodes the latent into a small feature map, upsamples
  it to the canvas, and convolves it down to one logit per cell, so the "where to
  place" decision is spatial.

---

## Same as the paper

These are faithful to the method:

- **The framing.** Macro placement as sequential, one-macro-at-a-time Reinforcement Learning (RL), with the
  standard cells placed by a force-directed method and the result scored by a
  proxy.
- **The three-term proxy reward.** Wirelength, congestion, and density, combined
  with weights, delivered as a sparse reward at the end of the episode.
- **Action masking.** Illegal placements are forbidden by a mask rather than
  learned.
- **The learning algorithm.** Proximal Policy Optimization (PPO), actor-critic,
  with a value head estimating the return.
- **The Graph Neural Network idea.** Encoding the netlist as a graph so the policy
  reasons over connectivity, and (in the spatial-head versions) decoding to a
  per-cell distribution over the grid. This is the paper's central architectural
  bet, and it is the same bet here.

## Differs from the paper

These are deliberate simplifications for a laptop-scale reimplementation. Each row
says what the paper does, what this project does, and why.

| Aspect | Paper | This project | Why |
| --- | --- | --- | --- |
| Scale | Real netlists, hundreds of macros (for example Ariane, tensor-processing-unit blocks) | 4 to 14 macros, synthetic problems | Fits on a Central Processing Unit (CPU) in minutes; keeps every result verifiable |
| Standard cells | Millions of cells clustered (hMETIS partitioning) into thousands of soft clusters | A handful (12 to 20) of predefined abstract cluster nodes | Removes the clustering pipeline; keeps the macro-plus-soft-cluster reward structure |
| Grid | Coarse grid up to 128x128, size chosen automatically | Fixed 25x25 (30x30 for the hero demo) | Small enough to train fast and visualize |
| Congestion model | Routing-based congestion proxy | Rectangular Uniform wire DensitY (RUDY) estimate | RUDY is a standard, much simpler stand-in |
| Density handling | Enforced during placement by masking over-dense cells | In the reward only; the mask enforces overlap and fit, not density | Simpler mask; density stays a soft objective |
| Macro ordering | Heuristic ordering (for example by descending size) | Fixed netlist-index order | One fewer moving part |
| Graph Neural Network | A custom edge-based encoder producing node, edge, and graph embeddings | Two standard graph-convolution layers with a symmetrically-normalized adjacency | A common, well-understood message-passing encoder |
| Spatial policy head | Deconvolution (transposed-convolution) layers to a per-cell distribution | Feature map, bilinear upsample, then convolution to per-cell logits | Same spatial output, simpler layers |
| Generalization strategy | Pretrain on a corpus of blocks, then transfer and fine-tune to a new chip | Train per run on small random problems; no pretraining corpus | The corpus and transfer pipeline are out of scope at this scale |
| Compute | Distributed Proximal Policy Optimization (PPO) on a large accelerator cluster, many parallel workers, hours | Single environment; policy on a Graphics Processing Unit (GPU) from V4 on; minutes | Accessible hardware only |

## What the differences mean for the results

Two of these differences matter most, and they explain the honest findings in
[RESULTS.md](RESULTS.md):

- **No pretraining corpus, and tiny scale.** The paper's headline benefit is
  generalization: a policy pretrained across many blocks that transfers to a new
  chip. This project trains on small synthetic problems with no corpus, so it can
  test whether generalization *emerges* but cannot reproduce the paper's transfer
  setup. At 4 to 14 macros there is also little transferable structure to learn and
  little headroom over a random-with-masking baseline, which is why an early
  "advantage grows with scale" result did not hold up in a second run, where run-to-run
  noise proved as large as the effect.
- **A cheaper, softer reward.** RUDY (Rectangular Uniform wire DensitY) congestion and coarse-grid density are looser
  proxies than the paper's routing-based model, and density here is a soft
  objective rather than a hard mask. A looser reward teaches more problem-specific
  quirks and fewer transferable regularities.

None of these differences is hidden or accidental. They are the cost of making the
method small enough to build from scratch, run on a laptop, and verify at every
step. Where that cost changed a conclusion, this repository says so.
</content>
</invoke>

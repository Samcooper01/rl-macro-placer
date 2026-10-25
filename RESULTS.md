# Results

The numbers from every version, in plain text. The illustrated write-up, with figures built
from the trained models, is the page at
[samcooper01.github.io/rl-macro-placer](https://samcooper01.github.io/rl-macro-placer/).

Every result compares the learned policy with random placement, which drops each macro on any
allowed cell, on the same problems. Lower cost is better. Cost is the weighted sum of
wirelength, congestion and density (0.1 × wirelength + 10 × congestion + 2 × density); V1 is
scored on wirelength alone.

## Summary

| Version | Tested on | Learned cost | Random average | Better than random by | Random layouts it beats |
| --- | --- | --- | --- | --- | --- |
| V1 | One fixed problem | 12 (wirelength) | 133 (wirelength) | 91 percent | All 2,000 |
| V2 | One fixed problem | 42.1 | 44.7 | 5.8 percent | 90 percent |
| V3 | 200 unseen problems | 44.1 | 45.2 | 2.6 percent | 70 percent |
| V4 | The same 200 | 44.1 | 45.2 | 2.4 percent | 70 percent |
| V5 | The same 200 | 44.3 | 45.2 | 2.1 percent | 67 percent |
| V6 | 40 unseen problems at each of 6 sizes | Flat and graph policies, 4 to 14 macros | | −0.9 to +2.7 percent | Within the noise |

## One fixed problem: the loop works (V1 and V2)

**V1**, scored on wirelength alone, packs connected macros tightly together. Its layout has a
wirelength of about 12, against an average of 133 over 2,000 random layouts; the best random
layout scores 57.

**V2**, scored the paper's way, has a lower total cost than 90 percent of 1,000 random layouts.
It gets there by trading: much shorter wires, the same congestion, and slightly worse density.

| Score (lower is better) | Random average | Learned | Learned beats |
| --- | --- | --- | --- |
| Wirelength | 117 | 82 | 98 percent of random |
| Congestion | 1.11 | 1.12 | 47 percent of random |
| Density | 10.9 | 11.3 | 19 percent of random |
| **Total cost** | **44.7** | **42.1** | **90 percent of random** |

The four macros hold more area than all twelve clusters combined (62 cells against 50), so
packing them together saves a lot of wire but crowds the busiest bins. The force-directed step
starts from a random seed and training always used the same one; rescored with 20 other seeds,
the learned layout still beats random 95 percent of the time.

## Unseen problems: a small gain, unchanged by the paper's architecture (V3 to V5)

V3, V4 and V5 were each tested on the same 200 unseen problems, against the same 100 random
layouts per problem.

| Score (lower is better) | Random average | V3 (flat) | V4 (graph) | V5 (graph, spatial head) |
| --- | --- | --- | --- | --- |
| Wirelength | 123 | 114 | 114 | 115 |
| Congestion | 1.14 | 1.13 | 1.14 | 1.13 |
| Density | 10.8 | 10.7 | 10.7 | 10.7 |
| **Total cost** | **45.2** | **44.1** | **44.1** | **44.3** |
| Share of random layouts beaten | | 70 percent | 70 percent | 67 percent |

V3's gain over random is real but small, and almost all of it is shorter wiring. It is not
memorizing: it scores about the same on 200 problems it trained on (44.2) as on unseen ones
(44.1). The graph network (V4) and the spatial head (V5) made no measurable difference: V4
beats V3 on 99 of the 200 problems, a coin flip, and V5 beats it on 92, within the margin of
error. Keeping the cheapest of 100 random layouts beats V3 on 191 of the 200 test problems, V4
on 191 and V5 on 197.

**Why: one fixed arrangement.** None of the three policies reads the problem much. On the 200
test problems, V4 put its first macro on the same cell every time, and its second macro on the
same cell in 153. V5 did the same in 194 and 195. Deleting all of the netlist's connections from
V4's input never changes its first move, and changes 366 of its 800 moves in total. Each policy
has learned one good general arrangement, not a way to adapt to the problem in front of it.

## Does the graph policy pull ahead with size? (V6)

V6 trains two policies at each size: the graph policy (V5's design) and a flat policy that gets
the same inputs but reads them as one long list. Every run gets 100,000 placements, and each
size is tested on 40 unseen problems, each compared with a single random layout.

Each number is the percentage by which the policy's average cost is lower than random
placement's. Positive means better than random, negative means worse.

| Macros | Flat, second run | Graph, second run | Flat, first run | Graph, first run |
| --- | --- | --- | --- | --- |
| 4 | +2.6 | +2.7 | +2.1 | +1.3 |
| 6 | +1.6 | +1.3 | | |
| 8 | +2.4 | +0.1 | +0.5 | +2.0 |
| 10 | +1.3 | +0.3 | | |
| 12 | +1.1 | +0.1 | | |
| 14 | −0.3 | −0.9 | +1.7 | +2.4 |

The first run showed the graph policy rising with size; the second run did not. Neither run is
precise enough to settle it. Comparing each problem with a single random layout moves each
number by up to about 1.1 points on its own, and training once per point adds at least as much:
at 4 macros, with identical settings and test problems, the graph policy scored 1.3 percent in
one run and 2.7 in the other. A fair test needs several training runs per point, many random
layouts per problem, and a training budget that grows with problem size.

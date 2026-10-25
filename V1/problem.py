"""The fixed macro-placement problem for V1.

A small, self-contained instance: a set of point macros on a square grid,
connected by nets. V1 trains on THIS ONE fixed problem to validate the loop
(can the agent learn to lower wirelength?). Generalization comes in v2.

The netlist below is two clusters, {0,1,2,3} and {4,5,6,7}, each internally
connected, with two cross-links. A good placement keeps each cluster compact
and the two clusters near each other: something you can eyeball in render().
"""

CANVAS = 20          # the grid is CANVAS x CANVAS cells
NUM_MACROS = 8       # macros are 1x1 points in V1 (real sizes come in v2)

# Each net is a list of macro indices that are electrically connected.
NETS = [
    [0, 1], [1, 2], [2, 3], [0, 3],   # cluster A
    [4, 5], [5, 6], [6, 7], [4, 7],   # cluster B
    [0, 4], [3, 7],                   # cross-links between the clusters
]

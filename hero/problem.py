"""Hero problem: a larger floorplan used only for the front-facing demo.

Same structure as the V2 experiment (hard macros placed by RL, soft clusters
settled by force-directed placement, reward over the whole placement), just at a
bigger scale so the README animation looks like a real floorplan rather than a
toy. This does not touch any of the numbered experiment folders.

  indices 0 .. NUM_MACROS-1                          -> hard macros (RL places)
  indices NUM_MACROS .. NUM_MACROS+NUM_CLUSTERS-1    -> soft cell clusters
"""

CANVAS = 30                 # CANVAS x CANVAS grid
NUM_MACROS = 8
NUM_CLUSTERS = 20

# Macro sizes (width, height) in grid units, varied for a realistic look.
MACRO_SIZES = [(4, 4), (5, 4), (4, 5), (3, 4),
               (4, 3), (5, 3), (3, 5), (4, 4)]

# Each cluster's area (in cells) -> used for the density term.
CLUSTER_AREAS = [6, 6, 5, 5, 5, 4, 4, 4, 4, 4,
                 3, 3, 3, 3, 3, 3, 2, 2, 2, 2]

M = NUM_MACROS
# Nets connect macros (0..7) and clusters (8..27).
NETS = [
    [0, 8, 9],            # each macro with its local clusters
    [1, 10, 11],
    [2, 12, 13],
    [3, 14, 15],
    [4, 16, 17],
    [5, 18, 19],
    [6, 20, 21],
    [7, 22, 23],
    [0, 1, 9],            # neighbouring macros share a cluster
    [2, 3, 13],
    [4, 5, 17],
    [6, 7, 21],
    [1, 2, 24],           # cross links across the design
    [3, 4, 25],
    [5, 6, 26],
    [7, 0, 27],
    [8, 9, 10, 11],       # cluster-cluster nets
    [12, 13, 14, 15],
    [16, 17, 18, 19],
    [20, 21, 22, 23],
    [24, 25, 26, 27],
    [0, 4, 12, 20],       # a couple of wide nets spanning the chip
    [2, 6, 16, 24],
]

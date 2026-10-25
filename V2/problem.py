"""V2 problem: hard macros + soft standard-cell clusters + a full netlist.

Objects are indexed uniformly:
  indices 0 .. NUM_MACROS-1                 -> hard macros (fixed size, RL places them)
  indices NUM_MACROS .. NUM_MACROS+NUM_CLUSTERS-1 -> soft cell clusters (force-directed)

Nets reference these object indices, so a net can connect macros to clusters,
clusters to clusters, etc. This mirrors the paper: RL places the macros, then the
standard-cell clusters are placed by a force-directed method, and the reward is
computed over the whole placement (wirelength + congestion + density).
"""

CANVAS = 25                 # CANVAS x CANVAS grid
NUM_MACROS = 4
NUM_CLUSTERS = 12

# Macro sizes (width, height) in grid units. Kept small relative to the canvas.
MACRO_SIZES = [(4, 4), (4, 4), (3, 5), (5, 3)]

# Each cluster's area (in cells) -> used for the density term.
CLUSTER_AREAS = [6, 6, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3]

M = NUM_MACROS
# Nets connect macros (0..3) and clusters (4..15).
NETS = [
    [0, 4, 5],          # macro 0 with its local clusters
    [0, 1, 6],          # two macros share a cluster
    [1, 7, 8],
    [2, 9, 10],
    [2, 3, 11],
    [3, 12, 13],
    [4, 5, 6],          # cluster-cluster nets
    [7, 8, 9],
    [10, 11, 12],
    [13, 14, 15],
    [0, 14],            # long macro-cluster links
    [3, 15],
    [1, 2, 5, 10],      # a wider net spanning the design
]

"""Graph types of report Table 2 and network sizes of report Table 1 (sec 5.1.1).

Python 3.9 syntax: imported by profilers in both the workspace and the sparserdiff env.
"""

from collections import namedtuple

# key: short name for --configs. name: report Table 2 name. table: report table with the results.
# old_name: name printed in the original MQP logs. n_edges: Table 2 M = min(5N, N(N-1)/2).
GraphConfig = namedtuple(
    "GraphConfig", "key name table old_name n_nodes n_edges dx de batch_size"
)

CONFIGS = [
    GraphConfig("molecular", "Molecular Graph", 3, "Molecular Graphs", 20, 100, 9, 4, 128),
    GraphConfig("small_social", "Small Social Network", 4, "Small Social Networks", 50, 250, 9, 4, 128),
    GraphConfig("medium", "Medium Graph", 5, "Medium Graphs", 100, 500, 9, 4, 128),
    GraphConfig("large", "Large Graph", 6, "Very Large Graphs", 200, 1000, 9, 4, 128),
    GraphConfig("complex_edge", "Complex Edge Types", 7, "Complex Edge Types", 50, 250, 9, 8, 128),
    GraphConfig("complex_node", "Complex Node Types", 8, "Complex Node Types", 50, 250, 20, 4, 128),
]
CONFIGS_BY_KEY = {c.key: c for c in CONFIGS}
CONFIGS_BY_NAME = {c.name: c for c in CONFIGS}
CONFIGS_BY_OLD_NAME = {c.old_name: c for c in CONFIGS}

# Smoke run: smallest graph type, small batch.
SMOKE_CONFIG_KEYS = ["molecular"]
SMOKE_BATCH_SIZE = 4
SMOKE_SEEDS = 1
SMOKE_STEPS = 2

# Report Table 1 (DiGress, SparseDiff, SparserDiff, MG-Diff backbones).
N_LAYERS = 5
HIDDEN_DIMS = {
    "dx": 256,
    "de": 64,
    "dy": 64,
    "n_head": 8,
    "dim_ffX": 256,
    "dim_ffE": 64,
    "dim_ffy": 64,
}
HIDDEN_MLP_DIMS = {"X": 128, "E": 64, "y": 128}

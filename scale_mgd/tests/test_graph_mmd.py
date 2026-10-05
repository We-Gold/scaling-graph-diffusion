"""Graph MMD metrics (Table 15). Includes the RBF tests from test_rbf_mmd.py."""

import math

import networkx as nx
import numpy as np

from scale_mgd.metrics.graph_mmd import _nx_to_pyg_batch, compute_all_metrics, rbf_mmd
from scale_mgd.pipeline.evaluate import triangle_mmd


def _graph(node_ids, edges):
    G = nx.Graph()
    G.add_nodes_from(node_ids)
    G.add_edges_from(edges)
    return G


def test_pyg_batch_non_contiguous_labels():
    batch = _nx_to_pyg_batch([_graph([3, 5, 7], [(3, 5), (5, 7)])])
    assert batch.x.shape[0] == 3
    assert batch.edge_index.max().item() < 3


def test_pyg_batch_mixed():
    graphs = [
        _graph([0, 1, 2, 3], [(0, 1), (2, 3)]),
        _graph([10, 20, 30], [(10, 20), (20, 30)]),
        _graph([1, 4], [(1, 4)]),
    ]
    batch = _nx_to_pyg_batch(graphs)
    assert batch.x.shape[0] == 9
    assert batch.edge_index.max().item() < 9


def test_rbf_mmd_gap_graphs_not_nan():
    ref = [_graph([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)]), _graph([5, 8, 11], [(5, 8), (8, 11)])]
    gen = [_graph([2, 7, 9], [(2, 7), (7, 9)]), _graph([0, 1], [(0, 1)])]
    assert not math.isnan(rbf_mmd(ref, gen))


def test_mmd_of_set_with_itself_is_zero():
    graphs = [nx.cycle_graph(n) for n in range(5, 12)] + [nx.path_graph(n) for n in range(4, 9)]
    res = compute_all_metrics(graphs, graphs)
    for k, v in res.items():
        assert abs(v) < 1e-6, k


def test_triangle_mmd_saturates():
    """Fig 9 triangle MMD: non-overlapping histograms give 2 - 2 e^-0.5 = 0.786939."""
    val = triangle_mmd(np.array([0.0, 0.0]), np.array([5.0, 5.0]))
    assert abs(val - (2 - 2 * math.exp(-0.5))) < 1e-4

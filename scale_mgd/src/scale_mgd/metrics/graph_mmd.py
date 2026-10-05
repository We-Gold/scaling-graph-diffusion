"""
Graph MMD metrics for report Table 15 (Degree, Cluster, Spectre, RBF MMD) and Fig. 9 (triangle MMD).

Kernels and graph statistics are ported from the GraphRNN / SPECTRE / DiGress evaluation code
(`dist_helper.py`, `spectre_utils.py`, MIT license). The RBF MMD uses a random-weight GIN
(3 layers, hidden 35, sum pooling) as in SparseDiff, implemented with PyG instead of DGL.
"""

import numpy as np
import networkx as nx
import concurrent.futures
from functools import partial
from scipy.linalg import eigvalsh

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# MMD kernel functions (ported from dist_helper.py)
# ============================================================================

def gaussian_tv(x, y, sigma=1.0):
    """Gaussian kernel using Total Variation distance."""
    support_size = max(len(x), len(y))
    x = x.astype(float)
    y = y.astype(float)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))
    dist = np.abs(x - y).sum() / 2.0
    return np.exp(-dist * dist / (2 * sigma * sigma))


def gaussian(x, y, sigma=1.0):
    """Gaussian kernel using L2 distance."""
    support_size = max(len(x), len(y))
    x = x.astype(float)
    y = y.astype(float)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))
    dist = np.linalg.norm(x - y, 2)
    return np.exp(-dist * dist / (2 * sigma * sigma))


def _kernel_parallel_worker(t):
    x, samples2, kernel = t
    d = 0
    for s2 in samples2:
        d += kernel(x, s2)
    return d


def disc(samples1, samples2, kernel, is_parallel=True, *args, **kwargs):
    """Discrepancy between two sample sets."""
    d = 0
    if not is_parallel:
        for s1 in samples1:
            for s2 in samples2:
                d += kernel(s1, s2, *args, **kwargs)
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for dist_val in executor.map(_kernel_parallel_worker, [
                (s1, samples2, partial(kernel, *args, **kwargs)) for s1 in samples1
            ]):
                d += dist_val
    if len(samples1) * len(samples2) > 0:
        d /= len(samples1) * len(samples2)
    else:
        d = 1e+6
    return d


def compute_mmd(samples1, samples2, kernel, is_hist=True, *args, **kwargs):
    """MMD between two sample sets."""
    if is_hist:
        samples1 = [s1 / (np.sum(s1) + 1e-6) for s1 in samples1]
        samples2 = [s2 / (np.sum(s2) + 1e-6) for s2 in samples2]
    return (disc(samples1, samples1, kernel, *args, **kwargs)
            + disc(samples2, samples2, kernel, *args, **kwargs)
            - 2 * disc(samples1, samples2, kernel, *args, **kwargs))


# ============================================================================
# Graph-level statistics -> histogram -> MMD
# ============================================================================

def _degree_worker(G):
    return np.array(nx.degree_histogram(G))


def degree_stats(graph_ref_list, graph_pred_list, is_parallel=True):
    """Compute MMD between degree distributions of two graph sets."""
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if G.number_of_nodes() > 0
    ]

    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(_degree_worker, graph_ref_list):
                sample_ref.append(deg_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(_degree_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)
    else:
        for G in graph_ref_list:
            sample_ref.append(np.array(nx.degree_histogram(G)))
        for G in graph_pred_list_remove_empty:
            sample_pred.append(np.array(nx.degree_histogram(G)))

    return compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)


def _spectral_worker(G, n_eigvals=-1):
    try:
        eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
    except Exception:
        eigs = np.zeros(G.number_of_nodes())
    if n_eigvals > 0:
        eigs = eigs[1:n_eigvals + 1]
    spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
    spectral_pmf = spectral_pmf / (spectral_pmf.sum() + 1e-6)
    return spectral_pmf


def spectral_stats(graph_ref_list, graph_pred_list, is_parallel=True, n_eigvals=-1):
    """Compute MMD between spectral (Laplacian eigenvalue) distributions."""
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if G.number_of_nodes() > 0
    ]

    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for sp in executor.map(_spectral_worker, graph_ref_list,
                                   [n_eigvals] * len(graph_ref_list)):
                sample_ref.append(sp)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for sp in executor.map(_spectral_worker, graph_pred_list_remove_empty,
                                   [n_eigvals] * len(graph_pred_list_remove_empty)):
                sample_pred.append(sp)
    else:
        for G in graph_ref_list:
            sample_ref.append(_spectral_worker(G, n_eigvals))
        for G in graph_pred_list_remove_empty:
            sample_pred.append(_spectral_worker(G, n_eigvals))

    return compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)


def _clustering_worker(param):
    G, bins = param
    clustering_coeffs = list(nx.clustering(G).values())
    hist, _ = np.histogram(clustering_coeffs, bins=bins, range=(0.0, 1.0), density=False)
    return hist


def clustering_stats(graph_ref_list, graph_pred_list, bins=100, is_parallel=True):
    """Compute MMD between clustering coefficient distributions."""
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if G.number_of_nodes() > 0
    ]

    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for hist in executor.map(_clustering_worker,
                                     [(G, bins) for G in graph_ref_list]):
                sample_ref.append(hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for hist in executor.map(_clustering_worker,
                                     [(G, bins) for G in graph_pred_list_remove_empty]):
                sample_pred.append(hist)
    else:
        for G in graph_ref_list:
            sample_ref.append(_clustering_worker((G, bins)))
        for G in graph_pred_list_remove_empty:
            sample_pred.append(_clustering_worker((G, bins)))

    return compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv, sigma=1.0 / 10)


# ============================================================================
# RBF MMD via random GIN (PyG, no DGL)
# ============================================================================

class _GINConvLayer(nn.Module):
    """Simple GIN convolution layer using PyG primitives."""

    def __init__(self, in_dim, out_dim, eps=0.0):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor(eps))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x, edge_index):
        """
        x: (N, D) node features
        edge_index: (2, E) edges
        """
        from torch_geometric.utils import scatter

        # Aggregate neighbor features
        row, col = edge_index
        agg = scatter(x[col], row, dim=0, dim_size=x.size(0), reduce="sum")

        # GIN update: (1 + eps) * x + agg
        out = (1 + self.eps) * x + agg
        out = self.mlp(out)
        return out


class RandomGINFeatureExtractor(nn.Module):
    """
    Random-weight GIN for graph-level feature extraction.

    Architecture matches SparseDiff: 3 layers, hidden_dim=35, sum pooling.
    Weights are NOT trained: used only for computing graph embeddings
    for RBF MMD evaluation.
    """

    def __init__(self, input_dim=1, hidden_dim=35, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(_GINConvLayer(input_dim, hidden_dim))
        # Subsequent layers
        for _ in range(num_layers - 1):
            self.layers.append(_GINConvLayer(hidden_dim, hidden_dim))

        self.eval()  # Always in eval mode: no training

    @torch.no_grad()
    def forward(self, x, edge_index, batch):
        """
        x: (N_total, D) node features
        edge_index: (2, E_total) edges
        batch: (N_total,) graph assignment

        Returns: (num_graphs, hidden_dim) graph embeddings
        """
        from torch_geometric.nn import global_add_pool

        for layer in self.layers:
            x = layer(x, edge_index)

        # Sum pooling per graph
        return global_add_pool(x, batch)


def _nx_to_pyg_batch(graph_list):
    """Convert a list of NetworkX graphs to a batched PyG representation."""
    from torch_geometric.data import Data, Batch

    data_list = []
    for G in graph_list:
        n = G.number_of_nodes()
        if n == 0:
            continue

        # Relabel nodes to contiguous 0..n-1 (subgraph extraction can leave gaps)
        G = nx.convert_node_labels_to_integers(G)

        # Node features: degree (normalized)
        degrees = torch.tensor([d for _, d in G.degree()], dtype=torch.float32)
        x = degrees.unsqueeze(-1)  # (n, 1)

        # Edges
        edges = list(G.edges())
        if len(edges) > 0:
            src = [e[0] for e in edges] + [e[1] for e in edges]
            dst = [e[1] for e in edges] + [e[0] for e in edges]
            edge_index = torch.tensor([src, dst], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        data_list.append(Data(x=x, edge_index=edge_index))

    if len(data_list) == 0:
        return None

    return Batch.from_data_list(data_list)


def rbf_mmd(graph_ref_list, graph_pred_list, device=None,
            sigma_range=None, seed=42):
    """
    Compute RBF kernel MMD between two sets of graphs using random GIN embeddings.

    Args:
        graph_ref_list: list of nx.Graph (reference)
        graph_pred_list: list of nx.Graph (generated)
        device: torch device (default: cpu)
        sigma_range: list of sigma values for RBF kernel
        seed: random seed for GIN weights

    Returns:
        float: max RBF MMD across all sigma values
    """
    if device is None:
        device = torch.device("cpu")

    if sigma_range is None:
        sigma_range = [0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]

    # Filter empty graphs
    ref_graphs = [G for G in graph_ref_list if G.number_of_nodes() > 0]
    gen_graphs = [G for G in graph_pred_list if G.number_of_nodes() > 0]

    if len(ref_graphs) == 0 or len(gen_graphs) == 0:
        return float("nan")

    # Convert to PyG batches
    ref_batch = _nx_to_pyg_batch(ref_graphs)
    gen_batch = _nx_to_pyg_batch(gen_graphs)

    if ref_batch is None or gen_batch is None:
        return float("nan")

    ref_batch = ref_batch.to(device)
    gen_batch = gen_batch.to(device)

    # Create random GIN
    torch.manual_seed(seed)
    gin = RandomGINFeatureExtractor(input_dim=1, hidden_dim=35, num_layers=3).to(device)

    # Get embeddings
    ref_emb = gin(ref_batch.x, ref_batch.edge_index, ref_batch.batch)  # (n_ref, 35)
    gen_emb = gin(gen_batch.x, gen_batch.edge_index, gen_batch.batch)  # (n_gen, 35)

    # Compute pairwise distances for sigma scaling
    ref_emb_np = ref_emb.cpu().numpy()
    gen_emb_np = gen_emb.cpu().numpy()
    all_emb = np.concatenate([ref_emb_np, gen_emb_np], axis=0)

    # Mean pairwise distance
    from scipy.spatial.distance import pdist
    if len(all_emb) > 1:
        mean_dist = np.mean(pdist(all_emb))
        if mean_dist < 1e-10:
            mean_dist = 1.0
    else:
        mean_dist = 1.0

    # Compute RBF MMD for each sigma
    max_mmd = 0.0
    n_ref = len(ref_emb_np)
    n_gen = len(gen_emb_np)

    for sigma_mult in sigma_range:
        sigma = sigma_mult * mean_dist

        def rbf_kernel(x, y):
            dist_sq = np.sum((x - y) ** 2)
            return np.exp(-dist_sq / (2 * sigma * sigma))

        # K(ref, ref)
        k_rr = 0.0
        for i in range(n_ref):
            for j in range(n_ref):
                k_rr += rbf_kernel(ref_emb_np[i], ref_emb_np[j])
        k_rr /= n_ref * n_ref

        # K(gen, gen)
        k_gg = 0.0
        for i in range(n_gen):
            for j in range(n_gen):
                k_gg += rbf_kernel(gen_emb_np[i], gen_emb_np[j])
        k_gg /= n_gen * n_gen

        # K(ref, gen)
        k_rg = 0.0
        for i in range(n_ref):
            for j in range(n_gen):
                k_rg += rbf_kernel(ref_emb_np[i], gen_emb_np[j])
        k_rg /= n_ref * n_gen

        mmd = k_rr + k_gg - 2 * k_rg
        max_mmd = max(max_mmd, mmd)

    return max_mmd


# ============================================================================
# Convenience: compute all metrics at once
# ============================================================================

def compute_all_metrics(graph_ref_list, graph_pred_list, skip_rbf=False, device=None):
    """
    Compute all SparseDiff-style graph generation metrics.

    Returns a dict with keys: degree_mmd, cluster_mmd, spectre_mmd, rbf_mmd.
    """
    results = {}

    print("  Computing Degree MMD...")
    results["degree_mmd"] = degree_stats(graph_ref_list, graph_pred_list)

    print("  Computing Clustering MMD...")
    results["cluster_mmd"] = clustering_stats(graph_ref_list, graph_pred_list)

    print("  Computing Spectral MMD...")
    results["spectre_mmd"] = spectral_stats(graph_ref_list, graph_pred_list)

    if skip_rbf:
        results["rbf_mmd"] = float("nan")
    else:
        print("  Computing RBF MMD (neural)...")
        results["rbf_mmd"] = rbf_mmd(graph_ref_list, graph_pred_list, device=device)

    return results

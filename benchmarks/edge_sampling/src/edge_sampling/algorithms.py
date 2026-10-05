"""Edge sampling methods for SparserDiff Experiment 1 (report Sec 4.2 and 5.2.1).

Function name to report name:

- ``materialize_adj_matrix_fast``: "Materialize Adjacency Matrix" (labelled "SparseDiff" in the CSV and fig 6).
- ``rejection_sampling``: "Rejection Sampling".
- ``novel_sampling_faster_floyd``: "Ours" (Algorithm 4: prefix array L, Floyd sampling, binary search).
- ``materialize_adj_matrix``: input graph generator for the benchmark. Not a benchmarked method.
"""

import numpy as np
import bisect


def sample_without_replacement_floyd(population_size, sample_size):
    """
    Uniformly sample `sample_size` distinct integers from [0, population_size)
    using Floyd's algorithm.

    Returns a NumPy array of sampled indices.
    """

    if sample_size < 0:
        raise ValueError("sample_size must be non-negative")
    if population_size < 0:
        raise ValueError("population_size must be non-negative")
    if sample_size > population_size:
        raise ValueError("sample_size cannot exceed population_size")

    selected = set()
    start = population_size - sample_size
    for j in range(start, population_size):
        t = np.random.randint(0, j + 1)
        if t in selected:
            selected.add(j)
        else:
            selected.add(t)

    return np.fromiter(selected, dtype=np.int64)

def make_edge_list_top_right(edge_list):
    """
    Takes an edge list, and assumes that it should be symmetric (undirected graph).
    Edges that are in the lower triangle are reflected to the upper triangle to make sure all edges are represented.
    Edges are not double counted, so if an edge (i, j) is in the edge list, it will only be represented as (i, j) if i < j, and not as (j, i).
    Edges on the diagonal are ignored.

    Parameters:
    - edge_list: A list of tuples, where each tuple contains the row and column indices of an edge in the adjacency matrix.

    Returns:
    - A Python list of tuples, where each tuple contains the row and column indices of the edges in the upper triangle of the adjacency matrix.
    """

    edge_set = set()
    for row, col in edge_list:
        if row < col:  # Only include edges in the upper triangle
            edge_set.add((row, col))
        elif col < row:  # Reflect edges from the lower triangle to the upper triangle
            edge_set.add((col, row))
        # Ignore edges on the diagonal (row == col)

    return list(edge_set)


# Input graph generator for the benchmark, not a benchmarked method. Keep unchanged so the CSV is reproducible.
def materialize_adj_matrix(edge_list, num_nodes, num_samples, seed=None):
    """
    Materializes an adjacency matrix, flattens it, and removes all the existing edges from it by referencing the edge list.

    It then samples `num_samples` random indices from the flattened adjacency matrix, which represent potential edges that could be added to the graph. The function returns these sampled indices as a NumPy array.

    Parameters:
    - edge_list: A list of tuples, where each tuple contains the row and column indices of an edge in the adjacency matrix.
    - num_nodes: The total number of nodes in the graph.
    - num_samples: The number of random indices to sample from the flattened adjacency matrix.
    - seed: An optional integer to set the random seed for reproducibility.

    Returns:
    - A Python list of tuples, where each tuple contains the row and column indices of the sampled edges in the adjacency matrix.
    """

    if seed is not None:
        np.random.seed(seed)

    # Convert edge list to upper triangle format
    edge_list_top_right = make_edge_list_top_right(edge_list)

    # Build list of possible (row, col) pairs for top right (col > row, no diagonal)
    possible_edges = [(row, col) for row in range(num_nodes) for col in range(row + 1, num_nodes)]

    # Remove existing edges from possible_edges
    possible_edges_set = set(possible_edges)
    existing_edges_set = set(edge_list_top_right)
    available_edges = list(possible_edges_set - existing_edges_set)

    if len(available_edges) < num_samples:
        raise ValueError("Not enough available edges to sample from.")

    # Sample random edges from the available edges
    sampled_edges = [available_edges[i] for i in np.random.choice(len(available_edges), size=num_samples, replace=False)]

    return sampled_edges


def materialize_adj_matrix_fast(edge_list, num_nodes, num_samples, seed=None):
    """
    Faster, lower-overhead variant of materialize_adj_matrix that keeps the same
    algorithmic behavior: enumerate all upper-triangle edges, drop existing ones,
    and sample without replacement. Uses vectorized NumPy operations and boolean
    masking.
    """

    if seed is not None:
        np.random.seed(seed)

    # Precompute row start offsets for the flattened upper triangle.
    offsets = np.empty(num_nodes + 1, dtype=np.int64)
    running = 0
    for u in range(num_nodes):
        offsets[u] = running
        running += num_nodes - u - 1
    offsets[num_nodes] = running

    total_edges = offsets[-1]
    if total_edges < num_samples:
        raise ValueError("Not enough available edges to sample from.")

    # Start with all edges available.
    mask = np.ones(total_edges, dtype=bool)

    # Mark existing edges as unavailable.
    for u, v in edge_list:
        if u == v:
            continue
        if u > v:
            u, v = v, u
        idx = offsets[u] + (v - u - 1)
        if 0 <= idx < total_edges:
            mask[idx] = False

    available = np.nonzero(mask)[0]
    if available.size < num_samples:
        raise ValueError("Not enough available edges to sample from.")

    chosen = np.random.choice(available, size=num_samples, replace=False)

    # Map flat indices back to (row, col).
    row_starts = offsets[:-1]
    rows = np.searchsorted(row_starts, chosen, side="right") - 1
    cols = rows + 1 + (chosen - row_starts[rows])

    return list(zip(rows.tolist(), cols.tolist()))

def rejection_sampling(edge_list, num_nodes, num_samples, seed=None):
    """
    Performs rejection sampling to sample edges from the adjacency matrix while avoiding existing edges.

    Parameters:
    - edge_list: A list of tuples, where each tuple contains the row and column indices of an edge in the adjacency matrix.
    - num_nodes: The total number of nodes in the graph.
    - num_samples: The number of random edges to sample.
    - seed: An optional integer to set the random seed for reproducibility.

    Returns:
    - A Python list of tuples, where each tuple contains the row and column indices of the sampled edges in the adjacency matrix.
    """

    if seed is not None:
        np.random.seed(seed)

    # Convert edge list to upper triangle format
    edge_list_top_right = make_edge_list_top_right(edge_list)
    existing_edges_set = set(edge_list_top_right)

    sampled_edges_set = set()
    while len(sampled_edges_set) < num_samples:
        row = np.random.randint(0, num_nodes)
        col = np.random.randint(0, num_nodes)

        if row == col:
            continue  # Ignore diagonal

        edge = (row, col) if row < col else (col, row)
        if edge not in existing_edges_set and edge not in sampled_edges_set:
            sampled_edges_set.add(edge)

    return list(sampled_edges_set)


def novel_sampling_faster_floyd(edge_list, num_nodes, num_samples, seed=None, r=None):
    """
    Space-efficient edge sampling (report Algorithm 4). Samples `num_samples` missing
    upper-triangle edges. Uses Floyd's algorithm for the random missing-edge indices
    when `r` is not provided, then maps each index to (u, v) with the prefix array L
    and binary search.
    """

    if seed is not None:
        np.random.seed(seed)

    # Build upper-triangle adjacency
    adj = [[] for _ in range(num_nodes)]
    for u, v in edge_list:
        if u == v:
            continue
        if u > v:
            u, v = v, u
        adj[u].append(v)

    for u in range(num_nodes):
        adj[u].sort()

    # Compute A and prefix L in O(n)
    A = [0] * num_nodes
    L = [0] * num_nodes

    running = 0
    for u in range(num_nodes):
        total_possible = num_nodes - 1 - u
        A[u] = total_possible - len(adj[u])
        L[u] = running
        running += A[u]

    total_missing = running

    if total_missing < num_samples:
        raise ValueError("Not enough missing edges to sample.")

    if r is None:
        r_indices = sample_without_replacement_floyd(total_missing, num_samples)
    else:
        r_indices = np.atleast_1d(r)

    sampled_edges = []
    f_cache = {}

    for r_val in r_indices:
        u = bisect.bisect_right(L, r_val) - 1
        j = r_val - L[u]

        if u not in f_cache:
            adj_u = adj[u]
            f_cache[u] = [
                adj_u[i] - (u + 1) - i
                for i in range(len(adj_u))
            ]

        f = f_cache[u]
        ip = bisect.bisect_right(f, j)
        v = (u + 1) + j + ip

        sampled_edges.append((u, v))

    return sampled_edges

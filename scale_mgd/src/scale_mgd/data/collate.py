"""
Dense dataset items -> sparse training batches (report sec. 4.1.1).

Each graph becomes a node list X (padded with PAD to N_max) and M_max edge slots (u, v, tau).
Real edges come first in upper-triangle order (u < v). Empty slots are NO_EDGE slots whose
u and v point to the virtual node at index N_max.
"""

import numpy as np
import torch
import torch.nn.functional as F


def max_true_edges_in_dataset(dataset):
    """Return the largest number of real (type > NO_EDGE) edges of any graph in the dataset."""
    max_edges = 0
    for idx in range(len(dataset)):
        E_numpy = dataset[idx][2]
        if E_numpy.size == 0:
            continue
        true_edges = int((np.triu(E_numpy, k=1) > 1).sum())
        max_edges = max(max_edges, true_edges)
    return max_edges


class TruncationCounter:
    """Counts graphs whose edge list is cut at M_max. Prints a warning the first time."""

    def __init__(self):
        self.graphs_cut = 0
        self._warned = False

    def add(self, n_edges, M_max):
        if n_edges > M_max:
            self.graphs_cut += 1
            if not self._warned:
                print(f"WARNING: a graph has {n_edges} edges > M_max = {M_max}; "
                      "extra edges are dropped in training (original behavior).")
                self._warned = True


def training_collate_fn(batch, N_max, M_max, node_pad, edge_pad, truncation=None):
    """
    Args:
        batch: list of (A_list, C_list, E_dense) dataset items
        N_max: node slots (dataset.max_length)
        M_max: edge slots
        node_pad: node type used for padding (dataset.types["PAD"])
        edge_pad: (type, u, v) for empty edge slots, normally (NO_EDGE, N_max, N_max)
        truncation: optional TruncationCounter
    Returns:
        X (B, N_max), E (B, M_max, 3), E_mask (B, M_max) bool for real edges, num_real_nodes (B,)
    """
    pad_type, pad_u, pad_v = edge_pad
    X_list, E_tensors, E_masks, num_nodes_list = [], [], [], []

    for item in batch:
        A_list, E_numpy = item[0], item[2]
        n_real = len(A_list)
        num_nodes_list.append(n_real)

        x_tens = torch.tensor(A_list, dtype=torch.long)
        if n_real < N_max:
            x_tens = F.pad(x_tens, (0, N_max - n_real), value=node_pad)
        else:
            x_tens = x_tens[:N_max]
        X_list.append(x_tens)

        # Upper-triangle real edges, row-major order (0 = MASK, 1 = NO_EDGE)
        rows, cols = np.nonzero(np.triu(E_numpy, k=1) > 1)
        edge_list = [(int(i), int(j), int(E_numpy[i, j])) for i, j in zip(rows, cols)]
        if truncation is not None:
            truncation.add(len(edge_list), M_max)

        e_tens = torch.empty((M_max, 3), dtype=torch.long)
        e_tens[:, 0] = pad_u
        e_tens[:, 1] = pad_v
        e_tens[:, 2] = pad_type
        e_mask = torch.zeros(M_max, dtype=torch.bool)
        for k, (u, v, t) in enumerate(edge_list[:M_max]):
            e_tens[k, 0] = u
            e_tens[k, 1] = v
            e_tens[k, 2] = t
            e_mask[k] = True

        E_tensors.append(e_tens)
        E_masks.append(e_mask)

    return (
        torch.stack(X_list),
        torch.stack(E_tensors),
        torch.stack(E_masks),
        torch.tensor(num_nodes_list, dtype=torch.long),
    )

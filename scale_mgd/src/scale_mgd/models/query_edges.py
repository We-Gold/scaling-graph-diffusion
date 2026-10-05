"""
Scale-MGD with query edges (the model used for the runs behind report Figs 8-9 and Tables 14-15).

In training mode, each forward pass samples a random subset of all node pairs (query edges, as in
SparseDiff, report sec. 3.2) and adds them to the encoder input as NO_EDGE triples. Predictions are
made only for the original M slots. In eval mode no query edges are added.

Note: sampling uses torch.randperm(n(n-1)/2), which is O(n^2) memory. Only the base model
(SparseGNNMaskedDiffusionModel) is profiled in report sec. 5.1.
"""

import torch

from .gnn import SparseGNNMaskedDiffusionModel


def _condensed_to_row_col(condensed_idx, n):
    """
    Convert condensed upper-triangle indices (row-major, k=1) of an n x n matrix to (row, col).

    row = n - 2 - floor(sqrt(-8*i + 4*n*(n-1) - 7) / 2 - 0.5)
    col = i - row*n + row*(row+1)/2 + row + 1
    """
    n_f = float(n)
    i_f = condensed_idx.float()
    discriminant = -8.0 * i_f + 4.0 * n_f * (n_f - 1.0) - 7.0
    discriminant = discriminant.clamp(min=0.0)
    row = (n_f - 2.0 - torch.floor(torch.sqrt(discriminant) / 2.0 - 0.5)).long()
    row = row.clamp(0, n - 2)
    col = condensed_idx - row * n + (row * (row + 1)) // 2 + row + 1
    col = col.clamp(0, n - 1)
    return row, col


def sample_query_edges(num_nodes, edge_fraction, device):
    """
    Sample a random subset of the upper-triangle node pairs.

    Returns query_u, query_v: (Q,) node indices with Q = max(1, int(edge_fraction * n(n-1)/2)).
    """
    n = num_nodes
    max_edges = n * (n - 1) // 2
    if max_edges == 0:
        return (torch.zeros(0, dtype=torch.long, device=device),
                torch.zeros(0, dtype=torch.long, device=device))

    num_to_sample = max(1, int(edge_fraction * max_edges))
    num_to_sample = min(num_to_sample, max_edges)

    if num_to_sample < max_edges:
        perm = torch.randperm(max_edges, device=device)[:num_to_sample]
    else:
        perm = torch.arange(max_edges, device=device)

    return _condensed_to_row_col(perm, n)


class SparseGNNQueryEdgesModel(SparseGNNMaskedDiffusionModel):
    """Scale-MGD model whose encoder also sees random query edges during training."""

    def __init__(
        self,
        num_node_types,
        num_edge_types,
        max_nodes,
        hidden_size=128,
        gnn_layers=4,
        gnn_iterations=5,
        decoder_layers=4,
        dropout=0.2,
        edge_fraction=0.1,
        query_edge_type=1,  # NO_EDGE type for query edges
    ):
        super().__init__(
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
            max_nodes=max_nodes,
            hidden_size=hidden_size,
            gnn_layers=gnn_layers,
            gnn_iterations=gnn_iterations,
            decoder_layers=decoder_layers,
            dropout=dropout,
        )
        self.edge_fraction = edge_fraction
        self.query_edge_type = query_edge_type

    def forward(self, X, E, t, num_real_nodes=None, mask_token_u=None, mask_token_v=None,
                mask_token_t=0, marginal_distributions=None):
        B, N = X.shape
        device = X.device

        if mask_token_u is None:
            mask_token_u = self.max_nodes
        if mask_token_v is None:
            mask_token_v = self.max_nodes

        t_emb = self.time_embedding(t)

        # Sample query edges and add them to the encoder input (training only)
        E_augmented = E
        if self.training:
            query_u, query_v = sample_query_edges(N, self.edge_fraction, device)
            Q = query_u.shape[0]
            if Q > 0:
                query_triples = torch.stack([
                    query_u,
                    query_v,
                    torch.full((Q,), self.query_edge_type, device=device, dtype=torch.long),
                ], dim=-1).unsqueeze(0).expand(B, -1, -1)  # (B, Q, 3)
                E_augmented = torch.cat([E, query_triples], dim=1)  # (B, M+Q, 3)

        h_nodes = self.gnn_encoder(
            X, E_augmented, t_emb, num_real_nodes,
            mask_token_u, mask_token_v, mask_token_t,
            marginal_distributions=marginal_distributions,
        )

        # Edge decoder on the original M slots only
        return self.decode_edges(E, h_nodes, t_emb)

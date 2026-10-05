"""
Scale-MGD network (report sec. 4.1.4, Algorithm 1, Fig. 3).

Random unmasking GNN encoder for nodes, then an O(M) edge decoder: each edge slot is updated
from its own embedding and the embeddings of its two endpoint nodes. A zero "virtual node" at
index max_nodes stands for MASK / PAD pointers. Four heads predict node types, u, v, and edge type.

Module attribute names are kept from the original code so that old checkpoints still load.
"""

import torch
import torch.nn as nn

from .encoder import RandomUnmaskingGNNEncoder, SinusoidalPositionEmbedding


class SparseGNNMaskedDiffusionModel(nn.Module):
    """Base Scale-MGD model: random unmasking GNN encoder + GNN edge decoder."""

    def __init__(
        self,
        num_node_types,
        num_edge_types,
        max_nodes,
        hidden_size=128,
        gnn_layers=3,
        gnn_iterations=5,
        decoder_layers=4,
        dropout=0.1,
    ):
        super().__init__()

        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.max_nodes = max_nodes
        self.hidden_size = hidden_size

        # Timestep embedding
        self.time_embedding = SinusoidalPositionEmbedding(hidden_size)

        # GNN encoder with random unmasking (L_enc = gnn_iterations, L_mp = gnn_layers)
        self.gnn_encoder = RandomUnmaskingGNNEncoder(
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
            max_nodes=max_nodes,
            hidden_size=hidden_size,
            num_layers=gnn_layers,
            num_iterations=gnn_iterations,
        )

        # Edge embeddings. u and v have max_nodes + 1 values (index max_nodes = MASK / PAD).
        emb_size = max_nodes + 1
        self.edge_u_embedding = nn.Embedding(emb_size, hidden_size)
        self.edge_v_embedding = nn.Embedding(emb_size, hidden_size)
        self.edge_type_embedding = nn.Embedding(num_edge_types, hidden_size)

        # Positional encoding for the M edge slots
        max_edge_len = max(4096, max_nodes * 10)
        self.edge_position_embedding = nn.Embedding(max_edge_len, hidden_size)

        # Edge decoder: each slot only looks at its own u and v nodes, so it is O(M).
        self.decoder_layers = nn.ModuleList(
            [GNNEdgeDecoderLayer(hidden_size, dropout) for _ in range(decoder_layers)]
        )

        # Output heads. +1 on u/v covers the MASK / PAD pointer at index max_nodes.
        self.head_x = nn.Linear(hidden_size, num_node_types)
        self.head_eu = nn.Linear(hidden_size, max_nodes + 1)
        self.head_ev = nn.Linear(hidden_size, max_nodes + 1)
        self.head_et = nn.Linear(hidden_size, num_edge_types)

    def decode_edges(self, E, h_nodes, t_emb):
        """Run the edge decoder on the M slots of E and return all four logits."""
        B, M = E.shape[0], E.shape[1]
        device = E.device

        Eu = E[:, :, 0].clamp(0, self.max_nodes)
        Ev = E[:, :, 1].clamp(0, self.max_nodes)
        Et = E[:, :, 2].clamp(0, self.num_edge_types - 1)

        h_edges = (
            self.edge_u_embedding(Eu)
            + self.edge_v_embedding(Ev)
            + self.edge_type_embedding(Et)
        )
        edge_positions = torch.arange(M, device=device).unsqueeze(0).expand(B, -1)
        h_edges = h_edges + self.edge_position_embedding(edge_positions)
        h_edges = h_edges + t_emb.unsqueeze(1)

        # Append a zero virtual node, so pointers equal to max_nodes gather a valid row.
        zeros = torch.zeros(B, 1, self.hidden_size, device=device)
        h_nodes_extended = torch.cat([h_nodes, zeros], dim=1)

        for layer in self.decoder_layers:
            h_edges = layer(h_edges, h_nodes_extended, Eu, Ev)

        logits_X = self.head_x(h_nodes)
        logits_Eu = self.head_eu(h_edges)
        logits_Ev = self.head_ev(h_edges)
        logits_Et = self.head_et(h_edges)
        return logits_X, logits_Eu, logits_Ev, logits_Et

    def forward(self, X, E, t, num_real_nodes, mask_token_u=None, mask_token_v=None,
                mask_token_t=0, marginal_distributions=None):
        """
        X: (B, N) node types, E: (B, M, 3) edge slots [u, v, type], t: (B,) timesteps,
        num_real_nodes: (B,). Returns logits of shape (B, N, Cx), (B, M, N+1), (B, M, N+1), (B, M, Ct).
        """
        if mask_token_u is None:
            mask_token_u = self.max_nodes
        if mask_token_v is None:
            mask_token_v = self.max_nodes

        t_emb = self.time_embedding(t)
        h_nodes = self.gnn_encoder(
            X, E, t_emb, num_real_nodes, mask_token_u, mask_token_v, mask_token_t,
            marginal_distributions=marginal_distributions,
        )
        return self.decode_edges(E, h_nodes, t_emb)


class GNNEdgeDecoderLayer(nn.Module):
    """
    Updates edge features from the features of the two endpoint nodes.
    h_e' = LayerNorm(h_e + MLP([h_e, h_u, h_v])). Cost O(M).
    """

    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, h_edges, h_nodes, Eu, Ev):
        """h_edges: (B, M, H), h_nodes: (B, N+1, H), Eu, Ev: (B, M) indices."""
        H = h_edges.shape[-1]

        def gather_nodes(indices):
            idx_expanded = indices.unsqueeze(-1).expand(-1, -1, H)
            return torch.gather(h_nodes, 1, idx_expanded)

        h_u = gather_nodes(Eu)
        h_v = gather_nodes(Ev)
        update = self.mlp(torch.cat([h_edges, h_u, h_v], dim=-1))
        return self.norm(h_edges + update)

"""
GNN encoder with random edge unmasking (report sec. 4.1.4, Algorithm 1, Fig. 3).

Masked edge slots are filled with random values, a GNN runs on the result, and node
embeddings are averaged over L_enc random unmasking trials. Memory is O(N + M).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal timestep embeddings."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class GNNMessagePassingLayer(MessagePassing):
    """
    Single GNN layer for message passing on partially masked edges.
    Used inside the random unmasking encoder (Algorithm 1).
    """

    def __init__(self, hidden_size):
        super().__init__(aggr='add')
        self.hidden_size = hidden_size

        # Message function
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2 + hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )

        # Update function
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: (N, H) node features
            edge_index: (2, E) edge indices
            edge_attr: (E, H) edge features
        Returns:
            (N, H) updated node features
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        """
        Compute messages from node j to node i.

        Args:
            x_i: (E, H) target node features
            x_j: (E, H) source node features
            edge_attr: (E, H) edge features
        Returns:
            (E, H) messages
        """
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_mlp(msg_input)

    def update(self, aggr_out, x):
        """
        Update node features based on aggregated messages.

        Args:
            aggr_out: (N, H) aggregated messages
            x: (N, H) current node features
        Returns:
            (N, H) updated node features
        """
        update_input = torch.cat([x, aggr_out], dim=-1)
        return self.update_mlp(update_input)


class RandomUnmaskingGNNEncoder(nn.Module):
    """
    GNN encoder with random edge unmasking (report Algorithm 1).

    Handles partially masked edges by:
    1. Randomly replacing MASK tokens with valid tokens
    2. Running GNN message passing
    3. Repeating L times and aggregating results
    """

    def __init__(
        self,
        num_node_types,
        num_edge_types,
        max_nodes,
        hidden_size,
        num_layers=3,
        num_iterations=5
    ):
        super().__init__()

        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.max_nodes = max_nodes
        self.hidden_size = hidden_size
        self.num_iterations = num_iterations

        # Embeddings
        self.node_embedding = nn.Embedding(num_node_types, hidden_size)
        self.edge_type_embedding = nn.Embedding(num_edge_types, hidden_size)

        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GNNMessagePassingLayer(hidden_size)
            for _ in range(num_layers)
        ])

        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size)
            for _ in range(num_layers)
        ])

    def random_unmask_edges(self, E, mask_token_u, mask_token_v, mask_token_t, marginal_distributions=None):
        """
        Replace MASK tokens in edge list with random valid tokens.

        Args:
            E: (B, M, 3) edge list
            mask_token_*: int, the mask token ID for each component
            marginal_distributions: dict, optional. Keys 'u', 'v', 't' mapped to probability tensors.
        Returns:
            E_unmasked: (B, M, 3) edge list with random unmasking
        """
        B, M, _ = E.shape
        device = E.device

        E_unmasked = E.clone()

        # Unmask u (source nodes)
        mask_u = (E[:, :, 0] == mask_token_u)
        if mask_u.any():
            if marginal_distributions and 'u' in marginal_distributions:
                probs = marginal_distributions['u'].to(device)
                random_u = torch.multinomial(probs, B*M, replacement=True).view(B, M)
            else:
                random_u = torch.randint(1, self.max_nodes, (B, M), device=device)
            E_unmasked[:, :, 0] = torch.where(mask_u, random_u, E[:, :, 0])

        # Unmask v (target nodes)
        mask_v = (E[:, :, 1] == mask_token_v)
        if mask_v.any():
            if marginal_distributions and 'v' in marginal_distributions:
                probs = marginal_distributions['v'].to(device)
                random_v = torch.multinomial(probs, B*M, replacement=True).view(B, M)
            else:
                random_v = torch.randint(1, self.max_nodes, (B, M), device=device)
            E_unmasked[:, :, 1] = torch.where(mask_v, random_v, E[:, :, 1])

        # Unmask t (edge types)
        mask_t = (E[:, :, 2] == mask_token_t)
        if mask_t.any():
            if marginal_distributions and 't' in marginal_distributions:
                probs = marginal_distributions['t'].to(device)
                random_t = torch.multinomial(probs, B*M, replacement=True).view(B, M)
            else:
                random_t = torch.randint(1, self.num_edge_types, (B, M), device=device)
            E_unmasked[:, :, 2] = torch.where(mask_t, random_t, E[:, :, 2])

        return E_unmasked

    def forward(self, X, E, t_emb, num_real_nodes, mask_token_u=38, mask_token_v=38, mask_token_t=0, marginal_distributions=None):
        """
        Encode nodes using random unmasking GNN.

        Args:
            X: (B, N) node types
            E: (B, M, 3) edge list with MASK tokens
            t_emb: (B, H) timestep embedding
            num_real_nodes: (B,) number of real nodes per graph
        Returns:
            H_nodes: (B, N, H) node embeddings aggregated over L iterations
        """
        B, N = X.shape
        M = E.shape[1]
        device = X.device

        # Collect node embeddings from multiple random unmaskings
        all_node_embeddings = []

        for iteration in range(self.num_iterations):
            # Random unmasking
            E_unmasked = self.random_unmask_edges(
                E, mask_token_u, mask_token_v, mask_token_t,
                marginal_distributions=marginal_distributions
            )

            # Process each graph in batch independently
            batch_node_features = []

            for b in range(B):
                n_real = num_real_nodes[b].item()

                # Get node features for this graph
                x_b = X[b, :n_real]
                h_x = self.node_embedding(x_b)  # (n_real, H)
                h_x = h_x + t_emb[b:b+1].expand(n_real, -1)  # Add time embedding

                # Build edge_index and edge_attr
                E_b = E_unmasked[b]
                Eu_b = E_b[:, 0].clamp(0, n_real - 1)
                Ev_b = E_b[:, 1].clamp(0, n_real - 1)
                Et_b = E_b[:, 2].clamp(0, self.num_edge_types - 1)


                edge_index = torch.stack([Eu_b, Ev_b], dim=0)  # (2, M)
                edge_attr = self.edge_type_embedding(Et_b)  # (M, H)

                # Message passing
                for layer, norm in zip(self.gnn_layers, self.layer_norms):
                    h_x_new = layer(h_x, edge_index, edge_attr)
                    h_x = norm(h_x + h_x_new)  # Residual connection

                # Pad to full size N
                h_x_padded = torch.zeros(N, self.hidden_size, device=device)
                h_x_padded[:n_real] = h_x

                batch_node_features.append(h_x_padded)

            # Stack batch
            H_nodes_iter = torch.stack(batch_node_features, dim=0)  # (B, N, H)
            all_node_embeddings.append(H_nodes_iter)

        # Aggregate over iterations (mean)
        H_nodes = torch.stack(all_node_embeddings, dim=0).mean(dim=0)  # (B, N, H)

        return H_nodes

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_model import GraphTransformer


class D2GraphTransformer(nn.Module):
    """Discrete-to-discrete graph transformer used by zinc250k training.

    Expects integer inputs:
      A: (batch, n) atom type indices
      C: (batch, n) charge indices
      E: (batch, n, n) edge type indices
      t: (batch, 1) normalized timestep

    Returns logits (without the MASK class) with shapes:
      A_out: (batch, atom_classes-1, n)
      C_out: (batch, charge_classes-1, n)
      E_out: (batch, edge_classes-1, n, n)
    """

    def __init__(
        self,
        atom_dim: int,
        charge_dim: int,
        edge_dim: int,
        num_layers: int = 6,
        d_model: int = 128,
        num_heads: int = 8,
        dff: int = 512,
        position: bool = True,
        dropout: float = 0.0,
        max_pos: int = 512,
        hidden_dims: dict = None,
        hidden_mlp_dims: dict = None,
    ):
        super().__init__()

        self.atom_dim = atom_dim
        self.charge_dim = charge_dim
        self.edge_dim = edge_dim
        self.pad_idx = 1

        # Embeddings for discrete inputs
        self.atom_emb = nn.Embedding(atom_dim, d_model)
        self.charge_emb = nn.Embedding(charge_dim, d_model)
        self.edge_emb = nn.Embedding(edge_dim, d_model)

        self.position = position
        if self.position:
            self.max_pos = max_pos
            self.register_buffer(
                "pos_table", self._build_sincos_position_embedding(max_pos, d_model)
            )

        # Build an underlying continuous graph transformer that produces node/edge representations
        input_dims = {"X": d_model, "E": d_model, "y": 1}

        if hidden_mlp_dims is None:
            hidden_mlp_dims = {"X": d_model, "E": d_model, "y": d_model}

        if hidden_dims is None:
            hidden_dims = {
                "dx": d_model,
                "de": d_model,
                "dy": d_model,
                "n_head": num_heads,
                "dim_ffX": dff,
                "dim_ffE": max(64, dff // 4),
                "dim_ffy": dff,
            }

        output_dims = {"X": d_model, "E": d_model, "y": d_model}

        self.encoder = GraphTransformer(
            n_layers=num_layers,
            input_dims=input_dims,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            output_dims=output_dims,
            dropout=dropout,
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU(),
        )

        # Prediction heads produce logits for classes excluding MASK (index 0)
        self.head_atom = nn.Linear(d_model, atom_dim - 1)
        self.head_charge = nn.Linear(d_model, charge_dim - 1)
        self.head_edge = nn.Linear(d_model, edge_dim - 1)

    def _build_sincos_position_embedding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(
        self,
        A: torch.LongTensor,
        C: torch.LongTensor,
        E: torch.LongTensor,
        t: torch.Tensor,
    ):
        # A: (b, n), C: (b, n), E: (b, n, n), t: (b, 1)
        b, n = A.size()

        node_mask = A != self.pad_idx

        X = self.atom_emb(A) + self.charge_emb(C)  # (b, n, d)
        if self.position:
            if n <= self.pos_table.size(1):
                X = X + self.pos_table[:, :n, :].to(X.device)
            else:
                # fallback to truncated sincos if sequence longer than max_pos
                pos = self._build_sincos_position_embedding(n, X.size(-1)).to(X.device)
                X = X + pos[:, :n, :]

        E_emb = self.edge_emb(E)  # (b, n, n, d)

        # y is timestep scalar in [0,1]
        y = t.float()

        out = self.encoder(X, E_emb, y, node_mask)
        X_out = out.X  # (b, n, d)
        E_out = out.E  # (b, n, n, d)

        # Apply heads -> (b, n, classes-1)
        A_logits = self.head_atom(X_out)
        C_logits = self.head_charge(X_out)
        E_logits = self.head_edge(E_out)  # (b, n, n, classes-1)

        # Permute to scheduler expected shapes
        A_logits = A_logits.permute(0, 2, 1).contiguous()
        C_logits = C_logits.permute(0, 2, 1).contiguous()
        E_logits = E_logits.permute(0, 3, 1, 2).contiguous()

        return A_logits, C_logits, E_logits, None

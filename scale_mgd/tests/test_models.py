"""Forward-pass shapes of the two Scale-MGD networks (report Alg. 1). From test_model_zoo.py."""

import pytest
import torch

from scale_mgd import SparseGNNMaskedDiffusionModel, SparseGNNQueryEdgesModel
from scale_mgd.models import MODEL_REGISTRY

B, N, M = 4, 38, 100  # ZINC-like
C_x, C_e = 10, 5
HIDDEN = 32


def _random_input():
    X = torch.randint(0, C_x, (B, N))
    E = torch.zeros(B, M, 3, dtype=torch.long)
    E[:, :, 0] = torch.randint(0, N + 1, (B, M))  # includes the MASK / PAD pointer N
    E[:, :, 1] = torch.randint(0, N + 1, (B, M))
    E[:, :, 2] = torch.randint(0, C_e, (B, M))
    t = torch.randint(0, 100, (B,))
    num_real = torch.randint(5, N + 1, (B,))
    return X, E, t, num_real


@pytest.mark.parametrize("name,kwargs", [("gnn", {}), ("query_edges", {"edge_fraction": 0.2})])
@pytest.mark.parametrize("train_mode", [False, True])
def test_forward_shapes(name, kwargs, train_mode):
    model = MODEL_REGISTRY[name](num_node_types=C_x, num_edge_types=C_e, max_nodes=N,
                                 hidden_size=HIDDEN, **kwargs)
    model.train(train_mode)
    X, E, t, num_real = _random_input()
    marg = {"t": torch.tensor([0.0, 0.5, 0.2, 0.2, 0.1])}
    logits_X, logits_Eu, logits_Ev, logits_Et = model(X, E, t, num_real, marginal_distributions=marg)
    assert logits_X.shape == (B, N, C_x)
    # Pointer heads have N + 1 classes (index N = MASK / PAD virtual node)
    assert logits_Eu.shape == (B, M, N + 1)
    assert logits_Ev.shape == (B, M, N + 1)
    assert logits_Et.shape == (B, M, C_e)
    for x in (logits_X, logits_Eu, logits_Ev, logits_Et):
        assert torch.isfinite(x).all()


def test_state_dict_keys_match_original_names():
    """Checkpoints from the MQP repo must load: module names are unchanged."""
    model = SparseGNNQueryEdgesModel(num_node_types=C_x, num_edge_types=C_e, max_nodes=N)
    keys = set(model.state_dict())
    for prefix in ["gnn_encoder.node_embedding.weight", "gnn_encoder.gnn_layers.0.message_mlp.0.weight",
                   "edge_u_embedding.weight", "edge_v_embedding.weight", "edge_type_embedding.weight",
                   "edge_position_embedding.weight", "decoder_layers.0.mlp.0.weight",
                   "head_x.weight", "head_eu.weight", "head_ev.weight", "head_et.weight"]:
        assert prefix in keys
    # Query-edge model and base model share the same parameters
    base = SparseGNNMaskedDiffusionModel(num_node_types=C_x, num_edge_types=C_e, max_nodes=N, gnn_layers=4)
    assert keys == set(base.state_dict())

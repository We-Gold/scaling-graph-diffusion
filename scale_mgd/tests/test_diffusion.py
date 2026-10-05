"""Forward process, VLB loss, and posterior sampler (report 4.1.2-4.1.6). From test_sparse_graph_diffusion.py."""

import pytest
import torch

from scale_mgd.diffusion import (
    SparseGraphDiffusionLoss,
    SparseGraphDiffusionSchedule,
    q_sample_sparse_graph,
    sample_posterior_sparse_graph,
)

NUM_NODE_TYPES, NUM_EDGE_TYPES, MAX_NODES, T = 11, 6, 38, 100
B, N, M = 2, 38, 100
MASK_X, MASK_UV, MASK_T = 0, MAX_NODES, 0


def _clean():
    X = torch.randint(2, NUM_NODE_TYPES, (B, N))
    E = torch.zeros(B, M, 3, dtype=torch.long)
    E[:, :, 0] = torch.randint(0, MAX_NODES, (B, M))
    E[:, :, 1] = torch.randint(0, MAX_NODES, (B, M))
    E[:, :, 2] = torch.randint(2, NUM_EDGE_TYPES, (B, M))
    return X, E


def _schedules(atomic):
    return SparseGraphDiffusionSchedule(T, NUM_NODE_TYPES, NUM_EDGE_TYPES, MAX_NODES, atomic_edges=atomic)


def _fully_masked(E):
    return ((E[:, :, 0] == MASK_UV) & (E[:, :, 1] == MASK_UV) & (E[:, :, 2] == MASK_T)).sum().item()


def test_atomic_masks_whole_edges():
    torch.manual_seed(0)
    X, E = _clean()
    t = torch.tensor([50, 50])
    _, E_atomic = q_sample_sparse_graph(X, E, t, _schedules(True))
    _, E_indep = q_sample_sparse_graph(X, E, t, _schedules(False))
    # Atomic mode masks whole (u, v, tau) tuples, so many more slots are fully masked
    assert _fully_masked(E_atomic) > 0
    assert _fully_masked(E_atomic) > 2 * _fully_masked(E_indep)


def test_little_noise_at_t0():
    X, E = _clean()
    X_t, E_t = q_sample_sparse_graph(X, E, torch.tensor([0, 0]), _schedules(True))
    assert (X_t == X).float().mean() > 0.99
    assert (E_t == E).float().mean() > 0.99


@pytest.mark.parametrize("atomic", [True, False])
def test_vlb_loss_finite_and_backprop(atomic):
    X, E = _clean()
    sched = _schedules(atomic)
    t = torch.tensor([25, 75])
    X_t, E_t = q_sample_sparse_graph(X, E, t, sched)
    # Model output format: (B, spatial, classes)
    logits_X = torch.randn(B, N, NUM_NODE_TYPES, requires_grad=True)
    logits_Eu = torch.randn(B, M, MAX_NODES + 1, requires_grad=True)
    logits_Ev = torch.randn(B, M, MAX_NODES + 1, requires_grad=True)
    logits_Et = torch.randn(B, M, NUM_EDGE_TYPES, requires_grad=True)
    loss_fn = SparseGraphDiffusionLoss(sched, mask_token_X=MASK_X, mask_token_uv=MASK_UV, mask_token_t=MASK_T)
    loss, parts = loss_fn.compute_loss(logits_X, logits_Eu, logits_Ev, logits_Et, X, E, X_t, E_t, t,
                                       torch.ones(B) / T)
    assert loss.shape == (B,)
    assert torch.isfinite(loss).all()
    loss.mean().backward()
    assert logits_Eu.grad is not None and torch.isfinite(logits_Eu.grad).all()
    assert set(parts) == {"loss_X", "loss_Eu", "loss_Ev", "loss_Et", "loss_total"}


def test_posterior_sampler_valid_indices():
    sched = _schedules(True)
    X_t = torch.full((B, N), MASK_X)
    E_t = torch.zeros(B, M, 3, dtype=torch.long)
    E_t[:, :, :2] = MASK_UV
    E_t[:, :, 2] = MASK_T
    # Half the slots unmasked so the u < v rule is exercised
    E_t[:, : M // 2, 0] = 3
    E_t[:, : M // 2, 1] = 7
    E_t[:, : M // 2, 2] = 2
    t = torch.tensor([10, 10])
    X_tm1, E_tm1 = sample_posterior_sparse_graph(
        torch.randn(B, N, NUM_NODE_TYPES), torch.randn(B, M, MAX_NODES + 1),
        torch.randn(B, M, MAX_NODES + 1), torch.randn(B, M, NUM_EDGE_TYPES),
        X_t, E_t, t, sched, temperature=0.5, mask_token_uv=MASK_UV, mask_token_t=MASK_T,
    )
    assert X_tm1.shape == (B, N) and E_tm1.shape == (B, M, 3)
    assert X_tm1.max() < NUM_NODE_TYPES
    u, v = E_tm1[:, :, 0], E_tm1[:, :, 1]
    assert u.max() <= MASK_UV and v.max() <= MASK_UV and u.min() >= 0
    real = (u != MASK_UV) & (v != MASK_UV)
    assert (u[real] < v[real]).all()
    # Atomic rule: slots masked in E_t stay fully masked
    masked_before = E_t[:, :, 2] == MASK_T
    assert (E_tm1[masked_before][:, 2] == MASK_T).all()

"""CPU smoke test: forward pass, one training step, and reverse steps with Table 1 sizes."""

import warnings

import pytest
import torch

from mg_diff import D2GraphTransformer, DiffusionTransformer

# Report Table 1 backbone sizes, as set in the 5.1 profiler.
HIDDEN_DIMS = {"dx": 256, "de": 64, "dy": 64, "n_head": 8, "dim_ffX": 256, "dim_ffE": 64, "dim_ffy": 64}
HIDDEN_MLP_DIMS = {"X": 128, "E": 64, "y": 128}

# Parameter counts of the original profiled code
# (origin/sparse-diff-new-algo:SparseDiff_repo/mg_diff), computed once during migration.
EXPECTED_PARAMS = {(9, 4): 3348235, (9, 8): 3350287, (20, 4): 3353878}


def build(a_classes, e_classes, c_classes=1, n=6, steps=3):
    model = D2GraphTransformer(
        atom_dim=a_classes,
        charge_dim=c_classes,
        edge_dim=e_classes,
        num_layers=5,
        d_model=256,
        num_heads=8,
        dff=256,
        position=True,
        dropout=0.0,
        hidden_dims=HIDDEN_DIMS,
        hidden_mlp_dims=HIDDEN_MLP_DIMS,
    )
    with warnings.catch_warnings():
        # c_classes=1 (dummy charge, as in the profiler) divides by N=0 in alpha_schedule.
        warnings.simplefilter("ignore", RuntimeWarning)
        return DiffusionTransformer(
            mask_id=0,
            a_classes=a_classes,
            c_classes=c_classes,
            e_classes=e_classes,
            model=model,
            diffusion_step=steps,
            max_length=n,
        )


def random_graphs(a_classes, c_classes, e_classes, b=2, n=6):
    A = torch.randint(0, a_classes, (b, n))
    C = torch.randint(0, c_classes, (b, n))
    E = torch.randint(0, e_classes, (b, n, n))
    E = torch.tril(E) + torch.tril(E, -1).transpose(1, 2)
    return A, C, E


@pytest.mark.parametrize("a_classes,e_classes", list(EXPECTED_PARAMS))
def test_param_count(a_classes, e_classes):
    sched = build(a_classes, e_classes)
    assert sum(p.numel() for p in sched.parameters()) == EXPECTED_PARAMS[(a_classes, e_classes)]


@pytest.mark.parametrize("c_classes", [1, 3])
def test_forward_train_sample(c_classes):
    torch.manual_seed(0)
    a, e, b, n = 9, 4, 2, 6
    sched = build(a, e, c_classes=c_classes, n=n)
    A, C, E = random_graphs(a, c_classes, e, b=b, n=n)

    # Model pass (as profiled: predict_start on one-hot log inputs)
    log_A = torch.log(torch.nn.functional.one_hot(A, a).float().clamp(min=1e-30)).permute(0, 2, 1)
    log_C = torch.log(torch.nn.functional.one_hot(C, c_classes).float().clamp(min=1e-30)).permute(0, 2, 1)
    log_E = torch.log(torch.nn.functional.one_hot(E, e).float().clamp(min=1e-30)).permute(0, 3, 1, 2)
    t = torch.randint(0, sched.num_timesteps, (b,))
    with torch.no_grad():
        pA, pC, pE = sched.predict_start(log_A, log_C, log_E, t)
    assert pA.shape == (b, a, n) and pC.shape == (b, c_classes, n) and pE.shape == (b, e, n, n)

    # One training step
    opt = torch.optim.AdamW(sched.parameters(), lr=1e-3)
    opt.zero_grad()
    loss = sched(A, C, E)[-1]
    assert torch.isfinite(loss)
    loss.backward()
    opt.step()

    # Reverse process (3 steps)
    out = sched.sample(b)
    assert len(out) == 1
    A_s, C_s, E_s = out[0]
    assert A_s.shape == (b, n) and C_s.shape == (b, n) and E_s.shape == (b, n, n)
    assert torch.equal(E_s, E_s.transpose(1, 2))

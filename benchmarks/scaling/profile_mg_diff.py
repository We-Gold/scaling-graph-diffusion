"""MG-Diff profiler (report 5.1, MG-Diff rows of tables 3-8).

Runs in the uv workspace:  uv run python -m profile_mg_diff
Source: origin/sparse-diff-new-algo:SparseDiff_repo/mg_diff/profile_mgdiff.py. Stage bodies unchanged.
The network and scheduler come from baselines/mg_diff (workspace member mg-diff).
"""

import time

import numpy as np
import torch
import torch.nn.functional as F
from mg_diff import D2GraphTransformer, DiffusionTransformer

from common import build_parser, get_memory_usage, main, set_all_seeds
from configs import HIDDEN_DIMS, HIDDEN_MLP_DIMS, N_LAYERS

DEFAULT_STEPS = 10


def run_profile_on_seed(
    seed, max_nodes, batch_size, name, diffusion_steps, Xdim_output, Edim_output, device
):
    set_all_seeds(seed)

    # Configuration
    a_classes = Xdim_output
    c_classes = 1  # Dummy charge
    e_classes = Edim_output

    hidden_dims = dict(HIDDEN_DIMS)
    hidden_mlp_dims = dict(HIDDEN_MLP_DIMS)

    # Model initialization
    model = D2GraphTransformer(
        atom_dim=a_classes,
        charge_dim=c_classes,
        edge_dim=e_classes,
        num_layers=N_LAYERS,
        d_model=256,
        num_heads=8,
        dff=256,
        position=True,
        dropout=0.0,
        hidden_dims=hidden_dims,
        hidden_mlp_dims=hidden_mlp_dims,
    ).to(device)

    # Scheduler initialization
    noise_scheduler = DiffusionTransformer(
        mask_id=0,
        a_classes=a_classes,
        c_classes=c_classes,
        e_classes=e_classes,
        model=model,
        diffusion_step=diffusion_steps,
        max_length=max_nodes,
    ).to(device)

    # Dummy Data
    A = torch.randint(0, a_classes, (batch_size, max_nodes)).to(device)
    C = torch.randint(0, c_classes, (batch_size, max_nodes)).to(device)
    E = torch.randint(0, e_classes, (batch_size, max_nodes, max_nodes)).to(device)
    E = torch.tril(E) + torch.tril(E, -1).transpose(1, 2)

    # Warmup
    with torch.no_grad():
        _ = noise_scheduler(A, C, E)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    metrics = {}

    # 1. Profiling Forward Diffusion Process (Noise Addition)
    try:
        start_time = time.time()
        t = torch.randint(0, diffusion_steps, (batch_size,), device=device).long()
        with torch.no_grad():
            _, _, _ = noise_scheduler._add_noise1(A, t, a_classes)
            _, _, _ = noise_scheduler._add_noise2(C, t, c_classes)
            _, _, _ = noise_scheduler._add_noise3(E, t, e_classes)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        ram, gpu = get_memory_usage()
        metrics["Noise Addition"] = {
            "time": end_time - start_time,
            "ram": ram,
            "gpu": gpu,
        }
    except Exception as e:
        print(f"\nError in Noise Addition: {e}")
        metrics["Noise Addition"] = {"time": np.nan, "ram": np.nan, "gpu": np.nan}

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Profiling Single Model Forward Pass
    try:
        log_A = torch.log(F.one_hot(A, a_classes).float().clamp(min=1e-30)).permute(
            0, 2, 1
        )
        log_C = torch.log(F.one_hot(C, c_classes).float().clamp(min=1e-30)).permute(
            0, 2, 1
        )
        log_E = torch.log(F.one_hot(E, e_classes).float().clamp(min=1e-30)).permute(
            0, 3, 1, 2
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start_time = time.time()
        with torch.no_grad():
            _ = noise_scheduler.predict_start(log_A, log_C, log_E, t)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        ram, gpu = get_memory_usage()
        metrics["Model Pass"] = {"time": end_time - start_time, "ram": ram, "gpu": gpu}
    except Exception as e:
        print(f"\nError in Model Pass: {e}")
        metrics["Model Pass"] = {"time": np.nan, "ram": np.nan, "gpu": np.nan}

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Profiling Training Step (Forward + Backward)
    try:
        optimizer = torch.optim.AdamW(noise_scheduler.parameters(), lr=1e-3)
        optimizer.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start_time = time.time()
        _, _, _, loss = noise_scheduler(A, C, E)
        loss.backward()
        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        ram, gpu = get_memory_usage()
        metrics["Training Step"] = {
            "time": end_time - start_time,
            "ram": ram,
            "gpu": gpu,
        }
    except Exception as e:
        print(f"\nError in Training Step: {e}")
        metrics["Training Step"] = {"time": np.nan, "ram": np.nan, "gpu": np.nan}

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. Profiling Backward Diffusion Process (Sampling)
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # sample(n) generates n graphs of max_length nodes (internal batches of 256)
        start_time = time.time()
        with torch.no_grad():
            noise_scheduler.sample(batch_size)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        ram, gpu = get_memory_usage()
        metrics["Backward Process"] = {
            "time": end_time - start_time,
            "ram": ram,
            "gpu": gpu,
        }
    except Exception as e:
        print(f"\nError in Backward Process: {e}")
        metrics["Backward Process"] = {"time": np.nan, "ram": np.nan, "gpu": np.nan}

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


if __name__ == "__main__":
    parser = build_parser(__doc__, DEFAULT_STEPS, "out/raw/mg_diff.csv")
    main("MG-Diff", run_profile_on_seed, parser.parse_args(), DEFAULT_STEPS)

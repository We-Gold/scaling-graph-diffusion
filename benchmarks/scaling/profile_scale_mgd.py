"""Scale-MGD profiler (report 5.1, Scale-MGD rows of tables 3-8).

Runs in the uv workspace:  uv run python -m profile_scale_mgd
Source: origin/sparse-diff-new-algo:SparseDiff_repo/wpi-graph-ai-mqp-25-26/masked-diff-graph/profile_md4_sparse.py.
Stage bodies unchanged. Default 20 diffusion steps: this is what produced the Scale-MGD rows of tables 3-8
(the other models used 10), so its Backward Process times cover 2x the steps.
"""

import time

import numpy as np
import torch
import torch.nn.functional as F
from scale_mgd import SparseGNNMaskedDiffusionModel

from common import build_parser, get_memory_usage, main, set_all_seeds

DEFAULT_STEPS = 20

# Base Scale-MGD model as profiled (no query edges)
HIDDEN_SIZE = 256
GNN_LAYERS = 3
GNN_ITERATIONS = 5
DECODER_LAYERS = 4
DROPOUT = 0.1


def run_profile_on_seed(
    seed, max_nodes, batch_size, name, diffusion_steps, Xdim_output, Edim_output, device
):
    set_all_seeds(seed)

    hidden_size = HIDDEN_SIZE
    num_node_types = Xdim_output
    num_edge_types = Edim_output

    model = SparseGNNMaskedDiffusionModel(
        num_node_types=num_node_types,
        num_edge_types=num_edge_types,
        max_nodes=max_nodes,
        hidden_size=hidden_size,
        gnn_layers=GNN_LAYERS,
        gnn_iterations=GNN_ITERATIONS,
        decoder_layers=DECODER_LAYERS,
        dropout=DROPOUT,
    )

    model.to(device)
    model.eval()

    num_edges = min(max_nodes * max_nodes, max_nodes * 5)
    if num_edges < 1:
        num_edges = 1

    X = torch.randint(0, num_node_types, (batch_size, max_nodes), device=device)
    E_indices = torch.randint(0, max_nodes, (batch_size, num_edges, 2), device=device)
    E_types = torch.randint(
        0, num_edge_types, (batch_size, num_edges, 1), device=device
    )
    E = torch.cat([E_indices, E_types], dim=2)

    t = torch.randint(0, diffusion_steps, (batch_size,), device=device).float()
    num_real_nodes = torch.full((batch_size,), max_nodes, device=device)

    # Warmup
    with torch.no_grad():
        _ = model(X, E, t, num_real_nodes)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    metrics = {}

    # Noise Addition
    try:
        start_time = time.time()
        with torch.no_grad():
            X_noisy = X.clone()
            E_noisy = E.clone()
            mask_mask_x = torch.rand_like(X.float()) < 0.1
            X_noisy[mask_mask_x] = 0

            mask_mask_e = torch.rand_like(E[:, :, 2].float()) < 0.1
            E_noisy[:, :, 2][mask_mask_e] = 0

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

    # Model Pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    try:
        start_time = time.time()
        with torch.no_grad():
            out = model(X, E, t, num_real_nodes)
        end_time = time.time()
        ram, gpu = get_memory_usage()
        metrics["Model Pass"] = {"time": end_time - start_time, "ram": ram, "gpu": gpu}
    except Exception as e:
        print(f"\nError in Model Pass: {e}")
        metrics["Model Pass"] = {"time": np.nan, "ram": np.nan, "gpu": np.nan}

    # Training Step
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    try:
        model.train()
        start_time = time.time()

        logits_X, logits_Eu, logits_Ev, logits_Et = model(X, E, t, num_real_nodes)

        loss_x = F.cross_entropy(logits_X.transpose(1, 2), X)
        loss_e = (
            F.cross_entropy(logits_Eu.transpose(1, 2), E[:, :, 0])
            + F.cross_entropy(logits_Ev.transpose(1, 2), E[:, :, 1])
            + F.cross_entropy(logits_Et.transpose(1, 2), E[:, :, 2])
        )

        loss = loss_x + loss_e
        loss.backward()

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

    # Backward Process
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    try:
        model.eval()
        start_time = time.time()

        with torch.no_grad():
            curr_X = X.clone()
            curr_E = E.clone()
            for i in range(diffusion_steps):
                curr_t = torch.full(
                    (batch_size,), diffusion_steps - 1 - i, device=device
                ).float()

                lx, leu, lev, let = model(curr_X, curr_E, curr_t, num_real_nodes)

                # Simple greedy decoding for profiling
                curr_X = torch.argmax(lx, dim=-1)

                new_Eu = torch.argmax(leu, dim=-1)
                new_Ev = torch.argmax(lev, dim=-1)
                new_Et = torch.argmax(let, dim=-1)

                curr_E = torch.stack([new_Eu, new_Ev, new_Et], dim=2)

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

    return metrics


if __name__ == "__main__":
    parser = build_parser(__doc__, DEFAULT_STEPS, "out/raw/scale_mgd.csv")
    main("Scale-MGD", run_profile_on_seed, parser.parse_args(), DEFAULT_STEPS)

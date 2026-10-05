"""DiGress profiler (report 5.1, DiGress rows of tables 3-8).

Runs in the sparserdiff env:  uv run --project ../../sparserdiff python -m profile_digress
Source: origin/sparse-diff-new-algo:SparseDiff_repo/src/profile_diffusion.py. Stage bodies unchanged.
"""

import time

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from common import build_parser, get_memory_usage, main, set_all_seeds
from configs import HIDDEN_DIMS, HIDDEN_MLP_DIMS, N_LAYERS
from digress.diffusion.distributions import DistributionNodes
from digress.diffusion.extra_features import DummyExtraFeatures
from digress.diffusion_model_discrete import DiscreteDenoisingDiffusion

DEFAULT_STEPS = 10


class DummyDatasetInfos:
    def __init__(self, max_nodes, input_dims, output_dims):
        self.max_n_nodes = max_nodes
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.nodes_dist = DistributionNodes({max_nodes: 1})
        self.node_types = torch.ones(input_dims["X"]) / input_dims["X"]


def run_profile_on_seed(
    seed, max_nodes, batch_size, name, diffusion_steps, Xdim_output, Edim_output, device
):
    set_all_seeds(seed)

    # Configuration
    ydim_output = 0

    # Input dims must match what the model expects.
    # With DummyExtraFeatures, extra features are 0.
    # But 'y' gets 't' appended, so +1.
    input_dims = {"X": Xdim_output, "E": Edim_output, "y": ydim_output + 1}
    output_dims = {"X": Xdim_output, "E": Edim_output, "y": ydim_output}

    dataset_infos = DummyDatasetInfos(max_nodes, input_dims, output_dims)

    cfg = OmegaConf.create(
        {
            "general": {"name": name, "log_every_steps": 50, "number_chain_steps": 50},
            "model": {
                "diffusion_steps": diffusion_steps,
                "lambda_train": [5, 0],
                "n_layers": N_LAYERS,
                "hidden_mlp_dims": dict(HIDDEN_MLP_DIMS),
                "hidden_dims": dict(HIDDEN_DIMS),
                "diffusion_noise_schedule": "cosine",
                "transition": "uniform",
            },
        }
    )

    # Initialize model
    model = DiscreteDenoisingDiffusion(
        cfg=cfg,
        dataset_infos=dataset_infos,
        train_metrics=None,
        sampling_metrics=None,
        visualization_tools=None,
        extra_features=DummyExtraFeatures(),
        domain_features=DummyExtraFeatures(),
    )
    model.to(device)
    model.eval()

    # Randomly generate data: X (bs, n, dx) one-hot, E (bs, n, n, de) one-hot, y (bs, 0), node_mask (bs, n)
    X_int = torch.randint(0, Xdim_output, (batch_size, max_nodes), device=device)
    X = F.one_hot(X_int, num_classes=Xdim_output).float()

    E_int = torch.randint(
        0, Edim_output, (batch_size, max_nodes, max_nodes), device=device
    )
    # Symmetrize E
    E_int = torch.triu(E_int) + torch.triu(E_int, 1).transpose(1, 2)
    E = F.one_hot(E_int, num_classes=Edim_output).float()

    y = torch.zeros((batch_size, ydim_output), device=device)
    node_mask = torch.ones((batch_size, max_nodes), device=device).bool()

    # Warmup
    with torch.no_grad():
        _ = model.apply_noise(X, E, y, node_mask)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    metrics = {}

    # Profile Forward Diffusion Process (Noise Addition)
    try:
        start_time = time.time()
        with torch.no_grad():
            noisy_data = model.apply_noise(X, E, y, node_mask)
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
        noisy_data = None  # Flag partial failure

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Profile Model Forward Pass (Single Step Denoising Prediction)
    try:
        # If noisy_data is missing, try to recreate it locally for this step
        if noisy_data is None:
            with torch.no_grad():
                noisy_data = model.apply_noise(X, E, y, node_mask)

        start_time = time.time()
        with torch.no_grad():
            extra_data = model.compute_extra_data(noisy_data)
            _ = model.forward(noisy_data, extra_data, node_mask)
        end_time = time.time()
        ram, gpu = get_memory_usage()
        metrics["Model Pass"] = {"time": end_time - start_time, "ram": ram, "gpu": gpu}
    except Exception as e:
        print(f"\nError in Model Pass: {e}")
        metrics["Model Pass"] = {"time": np.nan, "ram": np.nan, "gpu": np.nan}

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Profile Training Step
    try:
        start_time = time.time()
        model.train()
        with torch.enable_grad():
            # 1. Apply noise
            noisy_data_train = model.apply_noise(X, E, y, node_mask)
            # 2. Forward
            extra_data = model.compute_extra_data(noisy_data_train)
            pred = model.forward(noisy_data_train, extra_data, node_mask)
            # 3. Dummy Loss and Backward
            loss = pred.X.mean() + pred.E.mean()  # Dummy loss
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

    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Profile Backward Diffusion Process (Full Sampling)
    try:
        start_time = time.time()
        with torch.no_grad():
            num_nodes_tensor = torch.full(
                (batch_size,), max_nodes, device=device, dtype=torch.int
            )
            # number_chain_steps=1 avoids an IndexError (size 0); keep_chain=0 saves nothing
            model.sample_batch(
                batch_id=0,
                batch_size=batch_size,
                keep_chain=0,
                number_chain_steps=1,
                save_final=0,
                num_nodes=num_nodes_tensor,
            )
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
    parser = build_parser(__doc__, DEFAULT_STEPS, "out/raw/digress.csv")
    main("DiGress", run_profile_on_seed, parser.parse_args(), DEFAULT_STEPS)

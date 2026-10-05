"""SparseDiff and SparserDiff profiler (report 5.1, SparseDiff and SparserDiff rows of tables 3-8).

Runs in the sparserdiff env:
    uv run --project ../../sparserdiff python -m profile_sparsediff --sampler original   # SparseDiff
    uv run --project ../../sparserdiff python -m profile_sparsediff --sampler novel      # SparserDiff
Source: origin/sparse-diff-new-algo:SparseDiff_repo/SparseDiff/profile_sparsediff.py. Stage bodies unchanged,
except that the edge fraction (old: 0.5 hard-coded in the config and in 3 sample_query_edges calls) is now
--edge-fraction (default 0.1, report 5.1.1) and the sampler is chosen by model.use_novel_sampling.
"""

import functools
import time

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from sparse_diffusion.diffusion.distributions import DistributionNodes
from sparse_diffusion.diffusion.extra_features import DummyExtraFeatures
from sparse_diffusion.diffusion.sample_edges import sample_query_edges
from sparse_diffusion.diffusion.sample_edges_utils import get_computational_graph
from sparse_diffusion.diffusion_model_sparse import DiscreteDenoisingDiffusion
from sparse_diffusion.utils import PlaceHolder
from torch_geometric.data import Batch, Data

from common import build_parser, get_memory_usage, main, set_all_seeds
from configs import HIDDEN_DIMS, HIDDEN_MLP_DIMS, N_LAYERS

DEFAULT_STEPS = 10
DEFAULT_EDGE_FRACTION = 0.1
SAMPLERS = {"original": ("SparseDiff", False), "novel": ("SparserDiff", True)}


class DummyDatasetInfos:
    def __init__(self, max_nodes, input_dims, output_dims):
        self.max_n_nodes = max_nodes
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.nodes_dist = DistributionNodes({max_nodes: 1})
        self.num_node_types = input_dims.X
        self.num_edge_types = input_dims.E
        self.num_charge_types = input_dims.charge
        self.use_charge = output_dims.charge > 0

        # Needed for initialization
        self.node_types = torch.ones(input_dims.X) / input_dims.X
        self.edge_types = torch.ones(input_dims.E) / input_dims.E
        self.charge_types = (
            torch.ones(input_dims.charge) / input_dims.charge
            if input_dims.charge > 0
            else torch.zeros(0)
        )

        self.is_molecular = False

    def to_one_hot(self, data):
        one_hot_data = data.clone()
        one_hot_data.x = F.one_hot(data.x, num_classes=self.num_node_types).float()
        one_hot_data.edge_attr = F.one_hot(
            data.edge_attr, num_classes=self.num_edge_types
        ).float()

        if not self.use_charge:
            one_hot_data.charge = data.x.new_zeros((*data.x.shape[:-1], 0))
        else:
            one_hot_data.charge = F.one_hot(
                data.charge + 1, num_classes=self.num_charge_types
            ).float()

        return one_hot_data


def run_profile_on_seed(
    seed,
    max_nodes,
    batch_size,
    name,
    diffusion_steps,
    Xdim_output,
    Edim_output,
    device,
    edge_fraction,
    use_novel_sampling,
):
    set_all_seeds(seed)

    # Configuration
    ydim_output = 0
    charge_dim = 0

    input_dims = PlaceHolder(
        X=Xdim_output, E=Edim_output, y=ydim_output + 1, charge=charge_dim
    )
    output_dims = PlaceHolder(
        X=Xdim_output, E=Edim_output, y=ydim_output, charge=charge_dim
    )

    dataset_infos = DummyDatasetInfos(max_nodes, input_dims, output_dims)

    cfg = OmegaConf.create(
        {
            "general": {
                "name": name,
                "log_every_steps": 50,
                "number_chain_steps": 50,
                "skip": 1,
                "test_variance": 0.0,
            },
            "model": {
                "diffusion_steps": diffusion_steps,
                "lambda_train": [5, 0],
                "n_layers": N_LAYERS,
                "hidden_mlp_dims": dict(HIDDEN_MLP_DIMS),
                "hidden_dims": dict(HIDDEN_DIMS),
                "diffusion_noise_schedule": "cosine",
                "transition": "uniform",
                "use_charge": False,
                "sign_net": False,
                "sn_hidden_dim": 0,
                "scaling_layer": False,
                "output_y": False,
                "dropout": 0.1,
                "extra_features": "none",
                "num_eigenvectors": 8,
                "edge_fraction": edge_fraction,
                "use_novel_sampling": use_novel_sampling,
                "autoregressive": False,
            },
            "train": {"batch_size": batch_size},
        }
    )

    # Initialize model
    model = DiscreteDenoisingDiffusion(
        cfg=cfg,
        dataset_infos=dataset_infos,
        train_metrics=None,
        extra_features=DummyExtraFeatures(),
        domain_features=DummyExtraFeatures(),
        val_sampling_metrics=None,
        test_sampling_metrics=None,
    )
    model.visualization_tools = None
    model.to(device)
    model.eval()

    # Create dummy batch
    data_list = []
    for _ in range(batch_size):
        # Random graph
        x = torch.randint(0, Xdim_output, (max_nodes,))
        # Random sparse graph
        # Create random edges
        num_edges = int(max_nodes * 2)  # Average degree 2
        edge_index = torch.randint(0, max_nodes, (2, num_edges))
        edge_attr = torch.randint(0, Edim_output, (num_edges,))

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, n_nodes=max_nodes)
        data.charge = torch.zeros(max_nodes, dtype=torch.long)
        data.y = torch.zeros((1, 0))
        data_list.append(data)

    batch = Batch.from_data_list(data_list).to(device)

    # Warmup
    with torch.no_grad():
        batch_one_hot = dataset_infos.to_one_hot(batch)
        sparse_noisy_data = model.apply_sparse_noise(batch_one_hot)

        triu_query_edge_index, _ = sample_query_edges(
            num_nodes_per_graph=batch_one_hot.ptr.diff(), edge_proportion=edge_fraction
        )
        _, comp_edge_index, comp_edge_attr = get_computational_graph(
            triu_query_edge_index=triu_query_edge_index,
            clean_edge_index=sparse_noisy_data["edge_index_t"],
            clean_edge_attr=sparse_noisy_data["edge_attr_t"],
        )
        sparse_noisy_data["comp_edge_index_t"] = comp_edge_index
        sparse_noisy_data["comp_edge_attr_t"] = comp_edge_attr
        _ = model.forward(sparse_noisy_data)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    metrics = {}

    # Profile Forward Diffusion Process (Noise Addition)
    try:
        start_time = time.time()
        with torch.no_grad():
            batch_one_hot = dataset_infos.to_one_hot(batch)
            sparse_noisy_data = model.apply_sparse_noise(batch_one_hot)
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
        sparse_noisy_data = None  # Flag partial failure

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Profile Model Forward Pass (Single Step Denoising Prediction)
    try:
        if sparse_noisy_data is None:
            with torch.no_grad():
                batch_one_hot = dataset_infos.to_one_hot(batch)
                sparse_noisy_data = model.apply_sparse_noise(batch_one_hot)

        # We need to prepare comp_edge_index/attr for forward pass
        # We already have sparse_noisy_data from previous step
        # We need to sample query edges
        start_time = time.time()
        with torch.no_grad():
            triu_query_edge_index, _ = sample_query_edges(
                num_nodes_per_graph=batch_one_hot.ptr.diff(), edge_proportion=edge_fraction
            )
            # Query edge sampling and the computational graph are part of SparseDiff's forward pass

            _, comp_edge_index, comp_edge_attr = get_computational_graph(
                triu_query_edge_index=triu_query_edge_index,
                clean_edge_index=sparse_noisy_data["edge_index_t"],
                clean_edge_attr=sparse_noisy_data["edge_attr_t"],
            )
            sparse_noisy_data["comp_edge_index_t"] = comp_edge_index
            sparse_noisy_data["comp_edge_attr_t"] = comp_edge_attr
            _ = model.forward(sparse_noisy_data)

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
            # 1. To one hot
            batch_one_hot = dataset_infos.to_one_hot(batch)
            # 2. Apply noise
            sparse_noisy_data_train = model.apply_sparse_noise(batch_one_hot)
            # 3. Query edges and computational graph
            triu_query_edge_index, _ = sample_query_edges(
                num_nodes_per_graph=batch_one_hot.ptr.diff(), edge_proportion=edge_fraction
            )
            _, comp_edge_index, comp_edge_attr = get_computational_graph(
                triu_query_edge_index=triu_query_edge_index,
                clean_edge_index=sparse_noisy_data_train["edge_index_t"],
                clean_edge_attr=sparse_noisy_data_train["edge_attr_t"],
            )
            sparse_noisy_data_train["comp_edge_index_t"] = comp_edge_index
            sparse_noisy_data_train["comp_edge_attr_t"] = comp_edge_attr
            # 4. Forward
            pred = model.forward(sparse_noisy_data_train)
            # 5. Backward (Dummy loss)
            loss = pred.node.mean() + pred.edge_attr.mean()
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
            # sample_batch expects num_nodes list/tensor
            num_nodes = torch.full(
                (batch_size,), max_nodes, device=device, dtype=torch.long
            )
            model.sample_batch(
                batch_id=0,
                batch_size=batch_size,
                keep_chain=0,
                number_chain_steps=1,
                save_final=0,
                num_nodes=num_nodes,
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
    parser = build_parser(__doc__, DEFAULT_STEPS, None)
    parser.add_argument(
        "--sampler",
        choices=sorted(SAMPLERS),
        required=True,
        help="original = SparseDiff, novel = SparserDiff (model.use_novel_sampling)",
    )
    parser.add_argument("--edge-fraction", type=float, default=DEFAULT_EDGE_FRACTION)
    args = parser.parse_args()
    label, novel = SAMPLERS[args.sampler]
    if args.out is None:
        args.out = f"out/raw/{label.lower()}.csv"
    fn = functools.partial(
        run_profile_on_seed, edge_fraction=args.edge_fraction, use_novel_sampling=novel
    )
    main(label, fn, args, DEFAULT_STEPS, edge_fraction=args.edge_fraction)

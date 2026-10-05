"""
End-to-end runs used by `scripts/run.py`, `scripts/regen_plots.py`, `scripts/reference_mmd.py`.

Ported from `framework/tests/experiment_architecture_comparison.py` and
`framework/tests/experiment_regen_plots.py` (one config and one dataset per call).
"""

import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from ..config import Config, load_config, package_root, save_config, seed_everything, select_device
from ..data import DATASETS
from ..metrics.graph_mmd import compute_all_metrics
from ..models import MODEL_REGISTRY
from .checkpoint import LoadCheckpointStep
from .evaluate import MoleculeValidityStep, SparseDiffMetricsStep, TriangleEvalStep
from .plots import EdgeCountPlotStep
from .runner import PipelineRunner
from .sample import dataset_to_nx, make_alphas, make_schedules
from .train import FullTrainingStep

RESULT_KEYS = ["degree_mmd", "cluster_mmd", "spectre_mmd", "rbf_mmd"]


def load_datasets(name, splits=("train", "test")):
    cls = DATASETS[name]
    return {split: cls(stage=split) for split in splits}


def build_model(cfg: Config, dataset, device):
    m = cfg.model
    kwargs = dict(
        num_node_types=len(dataset.types),
        num_edge_types=len(dataset.bonds),
        max_nodes=dataset.max_length,
        hidden_size=m.hidden_size,
        gnn_layers=m.gnn_layers,
        gnn_iterations=m.gnn_iterations,
        decoder_layers=m.decoder_layers,
        dropout=m.dropout,
    )
    if m.name == "query_edges":
        kwargs["edge_fraction"] = m.edge_fraction
    return MODEL_REGISTRY[m.name](**kwargs).to(device)


def _base_context(cfg, dataset, device, M_max):
    ctx = {
        "cfg": cfg,
        "dataset": dataset,
        "device": device,
        "M_max": M_max,
        "alphas": make_alphas(cfg.diffusion, device),
        "rng": np.random.default_rng(cfg.seed),
    }
    if cfg.diffusion.loss == "vlb" or cfg.eval.sampler == "posterior" \
            or cfg.plots.edge_count_sampler == "posterior":
        ctx["schedules"] = make_schedules(cfg.diffusion, dataset, device)
    return ctx


def default_run_dir(cfg: Config, dataset_name: str) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return package_root() / "outputs" / cfg.name / f"{dataset_name}_{stamp}"


def run_experiment(cfg: Config, dataset_name: str, run_dir=None, datasets=None):
    """Train + evaluate one dataset. Writes checkpoints/, eval/, viz/, config.yaml, result.json, results.csv."""
    if dataset_name not in cfg.datasets:
        raise KeyError(f"Dataset '{dataset_name}' has no entry under `datasets` in config '{cfg.name}'")
    ds_cfg = cfg.datasets[dataset_name]
    seed_everything(cfg.seed)
    device = select_device(cfg.device)
    run_dir = Path(run_dir) if run_dir else default_run_dir(cfg, dataset_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    with open(run_dir / "run_info.json", "w") as f:
        json.dump({"dataset": dataset_name}, f)
    print(f"Run dir: {run_dir}\nDevice: {device}")

    datasets = datasets or load_datasets(dataset_name)
    train_ds, test_ds = datasets["train"], datasets.get("test")
    print(f"Split sizes: train {len(train_ds)}, test {len(test_ds) if test_ds is not None else 0}")

    model = build_model(cfg, train_ds, device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model {cfg.model.name}: {param_count:,} parameters")

    ctx = _base_context(cfg, train_ds, device, ds_cfg.M_max)
    ctx["model"] = model
    ctx["dataset_test"] = test_ds

    is_molecule = dataset_name == "zinc250k"
    eval_dir, viz_dir = run_dir / "eval", run_dir / "viz"
    steps = [
        FullTrainingStep(cfg, ds_cfg.batch_size, ds_cfg.train_fraction, run_dir / "checkpoints"),
        SparseDiffMetricsStep(eval_dir, store_raw_tensors=is_molecule),
    ]
    if is_molecule:
        steps.append(MoleculeValidityStep(eval_dir))
    steps += [
        EdgeCountPlotStep(viz_dir / "edge_count_dist.png", cfg.plots.edge_count_num_samples,
                          sampler=cfg.plots.edge_count_sampler, temperature=cfg.eval.temperature),
        TriangleEvalStep(eval_dir),
    ]

    t0 = time.time()
    PipelineRunner(steps).run(ctx)
    elapsed = time.time() - t0

    # Same keys as the original results.csv / result.json
    tr = ctx.get("sparsediff_metrics_train", {})
    te = ctx.get("sparsediff_metrics_test", {})
    result = {
        "architecture": f"gnn_{cfg.model.name}" if cfg.model.name != "gnn" else "gnn_base",
        "loss_mode": cfg.diffusion.loss,
        "dataset": dataset_name,
        "params": param_count,
        "elapsed_s": round(elapsed, 1),
    }
    for k in RESULT_KEYS:
        result[k] = tr.get(k, float("nan"))
    if te:
        for k in RESULT_KEYS:
            result[f"{k}_test"] = te.get(k, float("nan"))
    losses = ctx.get("loss_history") or []
    result["final_loss"] = losses[-1] if losses else float("nan")
    times = ctx.get("train_time_per_epoch") or []
    result["avg_epoch_s"] = round(sum(times) / max(len(times), 1), 2)
    result["peak_mem_mb"] = ctx.get("peak_memory_mb")
    if "mol_metrics" in ctx:
        result["mol_validity"] = ctx["mol_metrics"]["validity"]
        result["mol_uniqueness"] = ctx["mol_metrics"]["uniqueness"]
        result["mol_novelty"] = ctx["mol_metrics"]["novelty"]
    result["triangle_mmd"] = ctx.get("triangle_mmd", float("nan"))
    result["graphs_cut_at_M_max"] = ctx.get("graphs_cut_at_M_max", 0)

    with open(run_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    with open(run_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)
    print(f"Wrote {run_dir / 'result.json'}")
    return result, run_dir


def regen_plots(cfg: Config, dataset_name: str, checkpoint, out_dir, dataset=None):
    """Figs 8-9 in report style (figsize and sample count from cfg.plots) from a saved checkpoint."""
    ds_cfg = cfg.datasets[dataset_name]
    seed_everything(cfg.seed)
    device = select_device(cfg.device)
    dataset = dataset if dataset is not None else load_datasets(dataset_name, ("train",))["train"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = _base_context(cfg, dataset, device, ds_cfg.M_max)
    ctx["model"] = build_model(cfg, dataset, device)
    # The report figures use cfg.plots.regen_num_samples graphs for both plots
    cfg.eval.num_samples = cfg.plots.regen_num_samples
    figsize = tuple(cfg.plots.figsize)
    steps = [
        LoadCheckpointStep(checkpoint),
        EdgeCountPlotStep(out_dir / "edge_count_dist.png", cfg.plots.regen_num_samples,
                          sampler=cfg.plots.edge_count_sampler, temperature=cfg.eval.temperature,
                          figsize=figsize),
        TriangleEvalStep(out_dir, figsize=figsize),
    ]
    PipelineRunner(steps).run(ctx)
    print(f"Plots saved to {out_dir}")
    return out_dir


def regen_from_run_dir(run_dir, overrides=None):
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml", overrides)
    with open(run_dir / "run_info.json") as f:
        dataset_name = json.load(f)["dataset"]
    return regen_plots(cfg, dataset_name, run_dir / "checkpoints" / "model_final.pt",
                       run_dir / "viz" / "regen")


def reference_mmd(dataset_name, out_path, device_name="auto", seed=42, max_graphs=1000,
                  datasets=None):
    """Table 15 'Reference' rows: MMD between the train and test splits (default loaders)."""
    device = select_device(device_name)
    datasets = datasets or load_datasets(dataset_name)
    # The original code called np.random.seed(42) before subsampling (seed 42 kept here)
    rng = np.random.default_rng(seed)
    train_graphs = dataset_to_nx(datasets["train"], max_graphs, rng)
    test_graphs = dataset_to_nx(datasets["test"], max_graphs, rng)
    print(f"{dataset_name}: {len(train_graphs)} train vs {len(test_graphs)} test graphs")
    metrics = compute_all_metrics(train_graphs, test_graphs, device=device)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    for k, v in metrics.items():
        print(f"  {k:<12} {v:.6f}")
    return metrics


__all__ = [
    "build_model",
    "load_datasets",
    "reference_mmd",
    "regen_from_run_dir",
    "regen_plots",
    "run_experiment",
]

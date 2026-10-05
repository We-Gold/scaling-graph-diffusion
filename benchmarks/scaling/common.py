"""Shared protocol of the 5.1 profilers: seeds, memory probe, seed loop, log printer, CSV writer.

Each profiler only defines `run_profile_on_seed(seed, max_nodes, batch_size, name, diffusion_steps,
Xdim_output, Edim_output, device)`, which returns {stage: {"time", "ram", "gpu"}}, and calls `main`.
Python 3.9 syntax: also runs in the sparserdiff env.
"""

import argparse
import csv
import os
import platform
import traceback

import numpy as np
import psutil
import torch

from configs import (
    CONFIGS,
    CONFIGS_BY_KEY,
    SMOKE_BATCH_SIZE,
    SMOKE_CONFIG_KEYS,
    SMOKE_SEEDS,
    SMOKE_STEPS,
)

SEEDS = [10, 23, 27, 36, 42, 58, 61, 74, 89, 94]
STAGES = ["Noise Addition", "Model Pass", "Training Step", "Backward Process"]

CSV_COLUMNS = [
    "model",
    "config",
    "n_nodes",
    "n_edges",
    "dx",
    "de",
    "batch_size",
    "steps",
    "edge_fraction",
    "seed",
    "stage",
    "time_s",
    "gpu_mb",
    "ram_mb",
    "device",
    "gpu_name",
    "torch_version",
    "python_version",
]


def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.default_rng(seed)  # no-op for the global numpy RNG; kept from the original profilers


def get_memory_usage():
    """Process RSS (MB) and peak CUDA memory since the last reset (MB, 0 on CPU)."""
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / 1024 / 1024
    if torch.cuda.is_available():
        gpu_usage = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        gpu_usage = 0
    return ram_usage, gpu_usage


def nan_metrics():
    return {"time": np.nan, "ram": np.nan, "gpu": np.nan}


def print_results(seeds, specific_results):
    """Same text format as the original MQP logs, so new runs can be diffed against them."""
    print("\n\n=== Profiling Results ===")
    print("\n--- Specific Results per Seed ---")
    header = f"{'Seed':<6} | {'Category':<18} | {'Time (s)':<10} | {'RAM (MB)':<10} | {'GPU (MB)':<10}"
    print(header)
    print("-" * len(header))
    for seed in seeds:
        for cat in STAGES:
            vals = specific_results[seed][cat]
            print(
                f"{seed:<6} | {cat:<18} | {vals['time']:<10.4f} | {vals['ram']:<10.2f} | {vals['gpu']:<10.2f}"
            )
        print("-" * len(header))

    print("\n--- Aggregated Results (Mean ± Std) ---")
    header = f"{'Category':<18} | {'Time (s)':<20} | {'RAM (MB)':<20} | {'GPU (MB)':<20}"
    print(header)
    print("-" * len(header))
    for cat in STAGES:
        cols = {}
        for key in ("time", "ram", "gpu"):
            vals = [specific_results[s][cat][key] for s in seeds]
            vals = [v for v in vals if not np.isnan(v)]
            cols[key] = (np.mean(vals), np.std(vals)) if vals else (np.nan, np.nan)
        time_str = f"{cols['time'][0]:.4f} ± {cols['time'][1]:.4f}"
        ram_str = f"{cols['ram'][0]:.2f} ± {cols['ram'][1]:.2f}"
        gpu_str = f"{cols['gpu'][0]:.2f} ± {cols['gpu'][1]:.2f}"
        print(f"{cat:<18} | {time_str:<20} | {ram_str:<20} | {gpu_str:<20}")


def run_profile(model_label, run_profile_on_seed, cfg, batch_size, seeds, steps, edge_fraction, device):
    """Profile one graph type over all seeds. Returns CSV rows."""
    print(f"\n--- Profiling {cfg.name} ---")
    print(f"Max Nodes: {cfg.n_nodes}, Batch Size: {batch_size}, Xdim: {cfg.dx}, Edim: {cfg.de}")
    print(f"Profiling with {steps} steps over {len(seeds)} seeds: {seeds}")
    print(f"Device: {device}")

    specific_results = {}
    for seed in seeds:
        print(f"Running Seed {seed}...", end="\r")
        try:
            specific_results[seed] = run_profile_on_seed(
                seed, cfg.n_nodes, batch_size, cfg.name, steps, cfg.dx, cfg.de, device
            )
        except Exception as e:
            print(f"Failed Seed {seed}: {e}")
            traceback.print_exc()
            specific_results[seed] = {cat: nan_metrics() for cat in STAGES}

    print_results(seeds, specific_results)

    is_cuda = device.type == "cuda"
    gpu_name = torch.cuda.get_device_name(device) if is_cuda else ""
    rows = []
    for seed in seeds:
        for cat in STAGES:
            vals = specific_results[seed][cat]
            rows.append(
                {
                    "model": model_label,
                    "config": cfg.name,
                    "n_nodes": cfg.n_nodes,
                    "n_edges": cfg.n_edges,
                    "dx": cfg.dx,
                    "de": cfg.de,
                    "batch_size": batch_size,
                    "steps": steps,
                    "edge_fraction": "" if edge_fraction is None else edge_fraction,
                    "seed": seed,
                    "stage": cat,
                    "time_s": vals["time"],
                    "gpu_mb": vals["gpu"],
                    "ram_mb": vals["ram"],
                    "device": device.type,
                    "gpu_name": gpu_name,
                    "torch_version": torch.__version__,
                    "python_version": platform.python_version(),
                }
            )
    return rows


def write_csv(rows, path):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {path}")


def build_parser(description, default_steps, default_out):
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--configs",
        default="all",
        help="comma-separated keys from configs.py (" + ",".join(c.key for c in CONFIGS) + ") or 'all'",
    )
    p.add_argument("--seeds", default=None, help="number of seeds (first k of SEEDS) or a comma list")
    p.add_argument("--steps", type=int, default=None, help=f"diffusion steps (default {default_steps})")
    p.add_argument("--batch-size", type=int, default=None, help="default 128 (Table 2)")
    p.add_argument("--out", default=default_out, help="CSV output path")
    p.add_argument(
        "--smoke",
        action="store_true",
        help=f"tiny run: {SMOKE_CONFIG_KEYS}, batch {SMOKE_BATCH_SIZE}, {SMOKE_SEEDS} seed, {SMOKE_STEPS} steps",
    )
    return p


def parse_seeds(arg):
    if arg is None:
        return list(SEEDS)
    if "," in arg:
        return [int(s) for s in arg.split(",") if s]
    return SEEDS[: int(arg)]


def main(model_label, run_profile_on_seed, args, default_steps, edge_fraction=None):
    """Run the selected graph types and write one CSV."""
    if args.smoke:
        keys = SMOKE_CONFIG_KEYS if args.configs == "all" else args.configs.split(",")
        batch_size = args.batch_size or SMOKE_BATCH_SIZE
        seeds = parse_seeds(args.seeds or str(SMOKE_SEEDS))
        steps = args.steps or SMOKE_STEPS
    else:
        keys = [c.key for c in CONFIGS] if args.configs == "all" else args.configs.split(",")
        batch_size = args.batch_size
        seeds = parse_seeds(args.seeds)
        steps = args.steps or default_steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for key in keys:
        cfg = CONFIGS_BY_KEY[key]
        bs = batch_size or cfg.batch_size
        rows += run_profile(model_label, run_profile_on_seed, cfg, bs, seeds, steps, edge_fraction, device)
    write_csv(rows, args.out)
    return rows

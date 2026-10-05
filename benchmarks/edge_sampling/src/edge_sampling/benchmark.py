"""Edge sampling benchmark, SparserDiff Experiment 1 (report Sec 5.2, table 9, fig 6).

Times each sampler with time.perf_counter and records the tracemalloc peak (Python heap, KB).
Run with: python -m edge_sampling.benchmark --help
"""

from edge_sampling.algorithms import (
    materialize_adj_matrix,
    materialize_adj_matrix_fast,
    rejection_sampling,
    novel_sampling_faster_floyd,
)
import matplotlib.pyplot as plt
import numpy as np
import time
import tracemalloc
import csv
import argparse
from collections import defaultdict

# Table 9 graph sizes.
DEFAULT_SIZES = [400, 800, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]


def build_method_colors(method_names):
    fixed_colors = {
        "SparseDiff": "#1f77b4",
        "Rejection Sampling": "#ff7f0e",
        "Ours": "#9467bd",
    }

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    if not color_cycle:
        color_cycle = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

    method_colors = {}
    fallback_idx = 0
    for name in method_names:
        if name in fixed_colors:
            method_colors[name] = fixed_colors[name]
        else:
            method_colors[name] = color_cycle[fallback_idx % len(color_cycle)]
            fallback_idx += 1

    return method_colors


def parse_method_list(raw_value):
    if raw_value is None:
        return None
    return [part.strip() for part in raw_value.split(",") if part.strip()]


def parse_int_list(raw_value):
    if raw_value is None:
        return None
    return [int(part.strip()) for part in raw_value.split(",") if part.strip()]


def filter_results_by_methods(
    speed_mean, mem_mean, speed_std=None, mem_std=None,
    include_methods=None, exclude_methods=None,
):
    include_set = set(include_methods) if include_methods else None
    exclude_set = set(exclude_methods) if exclude_methods else set()

    all_methods = sorted(set(speed_mean.keys()) | set(mem_mean.keys()))
    selected = []
    for method in all_methods:
        if include_set is not None and method not in include_set:
            continue
        if method in exclude_set:
            continue
        selected.append(method)

    if not selected:
        raise ValueError("No methods selected for plotting after include/exclude filters.")

    filtered_speed_mean = {m: speed_mean[m] for m in selected if m in speed_mean}
    filtered_mem_mean = {m: mem_mean[m] for m in selected if m in mem_mean}
    filtered_speed_std = {m: speed_std[m] for m in selected if m in speed_std} if speed_std else None
    filtered_mem_std = {m: mem_std[m] for m in selected if m in mem_std} if mem_std else None
    return filtered_speed_mean, filtered_mem_mean, filtered_speed_std, filtered_mem_std

def run_scaling_benchmark(sizes, percent_filled, num_to_sample, seeds):
    # "SparseDiff" is the report's "Materialize Adjacency Matrix" baseline, not SparseDiff's own sampler.
    methods = [
        ("SparseDiff", materialize_adj_matrix_fast),
        ("Rejection Sampling", rejection_sampling),
        ("Ours", novel_sampling_faster_floyd)
    ]
    method_names = [m[0] for m in methods]
    # raw[method][size_idx] = list of per-seed observations
    raw_speed = {name: [[] for _ in sizes] for name in method_names}
    raw_mem   = {name: [[] for _ in sizes] for name in method_names}

    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- Seed {seed} ({seed_idx + 1}/{len(seeds)}) ---")
        for size_idx, num_nodes in enumerate(sizes):
            num_edges = (num_nodes * (num_nodes - 1)) // 2
            num_edges_to_fill = min(int((1 + percent_filled) * num_nodes), num_edges)
            percent_filled_actual = (num_edges_to_fill / num_edges) * 100

            edge_list = materialize_adj_matrix([], num_nodes, num_edges_to_fill, seed)
            missing_edges = max(0, num_edges - len(edge_list))
            num_edges_to_sample = min(num_to_sample, missing_edges)

            print(
                f"  {num_nodes} nodes, {num_edges_to_fill} edges filled, "
                f"{num_edges_to_sample} to sample, {percent_filled_actual:.2f}% filled."
            )

            for name, func in methods:
                tracemalloc.start()
                t0 = time.perf_counter()
                func(edge_list, num_nodes, num_edges_to_sample, seed)
                t1 = time.perf_counter()
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                raw_speed[name][size_idx].append(t1 - t0)
                raw_mem[name][size_idx].append(peak / 1024)

    speed_mean = {name: [float(np.mean(raw_speed[name][i])) for i in range(len(sizes))] for name in method_names}
    speed_std  = {name: [float(np.std(raw_speed[name][i], ddof=1)) if len(raw_speed[name][i]) > 1 else 0.0 for i in range(len(sizes))] for name in method_names}
    mem_mean   = {name: [float(np.mean(raw_mem[name][i])) for i in range(len(sizes))] for name in method_names}
    mem_std    = {name: [float(np.std(raw_mem[name][i], ddof=1)) if len(raw_mem[name][i]) > 1 else 0.0 for i in range(len(sizes))] for name in method_names}

    return speed_mean, speed_std, mem_mean, mem_std, raw_speed, raw_mem


def save_results_csv(path, sizes, raw_speed, raw_mem, seeds):
    """Write one row per (num_nodes, method, seed) triple."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["num_nodes", "method", "seed", "time_seconds", "peak_memory_kb"])
        for method in sorted(raw_speed.keys()):
            for size_idx, num_nodes in enumerate(sizes):
                for seed_idx, seed in enumerate(seeds):
                    writer.writerow([
                        num_nodes,
                        method,
                        seed,
                        raw_speed[method][size_idx][seed_idx],
                        raw_mem[method][size_idx][seed_idx],
                    ])


def load_results_csv(path):
    """Load CSV and aggregate per-seed rows into mean ± std.

    Supports both the legacy schema (no 'seed' column) and the new schema.
    Returns sizes, speed_mean, speed_std, mem_mean, mem_std.
    """
    # raw[method][num_nodes] = list of per-seed values
    raw_speed = defaultdict(lambda: defaultdict(list))
    raw_mem   = defaultdict(lambda: defaultdict(list))
    size_set  = set()

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"num_nodes", "method", "time_seconds", "peak_memory_kb"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError("Results file is missing required columns.")

        for row in reader:
            num_nodes = int(row["num_nodes"])
            method    = row["method"]
            raw_speed[method][num_nodes].append(float(row["time_seconds"]))
            raw_mem[method][num_nodes].append(float(row["peak_memory_kb"]))
            size_set.add(num_nodes)

    sizes = sorted(size_set)
    speed_mean, speed_std, mem_mean, mem_std = {}, {}, {}, {}
    for method in sorted(raw_speed.keys()):
        s_vals = [raw_speed[method][n] for n in sizes]
        m_vals = [raw_mem[method][n]   for n in sizes]
        speed_mean[method] = [float(np.mean(v)) for v in s_vals]
        speed_std[method]  = [float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for v in s_vals]
        mem_mean[method]   = [float(np.mean(v)) for v in m_vals]
        mem_std[method]    = [float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for v in m_vals]

    return sizes, speed_mean, speed_std, mem_mean, mem_std


def plot_results(
    sizes, speed_mean, mem_mean,
    speed_std=None, mem_std=None,
    output_prefix="sampling",
):
    method_names = sorted(set(speed_mean.keys()) | set(mem_mean.keys()))
    method_colors = build_method_colors(method_names)

    plt.figure(figsize=(6, 4.5))
    for name in method_names:
        if name in speed_mean:
            yerr = speed_std[name] if speed_std and name in speed_std else None
            plt.errorbar(
                sizes, speed_mean[name],
                yerr=yerr, fmt='o-', capsize=4,
                label=name, color=method_colors[name],
            )
    plt.xlabel('Number of Nodes')
    plt.ylabel('Time (s)')
    plt.title('Sampling Speed Scaling')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_speed_scaling.png', dpi=300)
    plt.savefig(f'{output_prefix}_speed_scaling.pdf')
    plt.close()

    plt.figure(figsize=(6, 4.5))
    for name in method_names:
        if name in mem_mean:
            yerr = mem_std[name] if mem_std and name in mem_std else None
            plt.errorbar(
                sizes, mem_mean[name],
                yerr=yerr, fmt='o-', capsize=4,
                label=name, color=method_colors[name],
            )
    plt.xlabel('Number of Nodes')
    plt.ylabel('Peak Memory (KB)')
    plt.title('Sampling Memory Scaling')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_memory_scaling.png', dpi=300)
    plt.savefig(f'{output_prefix}_memory_scaling.pdf')
    plt.close()


def scaling_test(
    results_file="sampling_scaling_results.csv",
    from_saved=False,
    output_prefix="sampling",
    include_methods=None,
    exclude_methods=None,
    seeds=None,
    sizes=None,
):
    if seeds is None:
        seeds = [42, 62, 123, 456, 789]

    if sizes is None:
        sizes = DEFAULT_SIZES
    # Fill multiplier, not a percentage: the input graph has (1 + 2.8) * n = 3.8 n edges (table 9).
    percent_filled = 2.8
    num_to_sample = 1000

    if from_saved:
        sizes, speed_mean, speed_std, mem_mean, mem_std = load_results_csv(results_file)
        print(f"Loaded saved benchmark results from {results_file}")
    else:
        speed_mean, speed_std, mem_mean, mem_std, raw_speed, raw_mem = run_scaling_benchmark(
            sizes=sizes,
            percent_filled=percent_filled,
            num_to_sample=num_to_sample,
            seeds=seeds,
        )
        save_results_csv(results_file, sizes, raw_speed, raw_mem, seeds)
        print(f"Saved benchmark results to {results_file}")

    speed_mean, mem_mean, speed_std, mem_std = filter_results_by_methods(
        speed_mean=speed_mean,
        mem_mean=mem_mean,
        speed_std=speed_std,
        mem_std=mem_std,
        include_methods=include_methods,
        exclude_methods=exclude_methods,
    )

    plot_results(
        sizes, speed_mean, mem_mean,
        speed_std=speed_std, mem_std=mem_std,
        output_prefix=output_prefix,
    )
    print(
        f"Saved plots: {output_prefix}_speed_scaling.pdf, "
        f"{output_prefix}_memory_scaling.pdf"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-saved",
        action="store_true",
        help="Load benchmark results from --results-file and only generate plots.",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default="sampling_scaling_results.csv",
        help="Path to CSV file for saving/loading benchmark results.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="sampling",
        help="Prefix for output plot filenames.",
    )
    parser.add_argument(
        "--include-methods",
        type=str,
        default=None,
        help="Comma-separated list of method names to include in plots.",
    )
    parser.add_argument(
        "--exclude-methods",
        type=str,
        default=None,
        help="Comma-separated list of method names to exclude from plots.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated list of integer seeds for repeated runs (default: 42,62,123,456,789).",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default=None,
        help="Comma-separated list of graph sizes (default: table 9 sizes). Ignored with --from-saved.",
    )
    args = parser.parse_args()

    scaling_test(
        results_file=args.results_file,
        from_saved=args.from_saved,
        output_prefix=args.output_prefix,
        include_methods=parse_method_list(args.include_methods),
        exclude_methods=parse_method_list(args.exclude_methods),
        seeds=parse_int_list(args.seeds),
        sizes=parse_int_list(args.sizes),
    )


if __name__ == "__main__":
    main()

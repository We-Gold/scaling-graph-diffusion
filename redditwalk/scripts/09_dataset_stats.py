#!/usr/bin/env python3
"""Step 09 (report 6.3, Fig 12, Table 16): structure statistics of the RedditWalk sets.

Plots degree and clustering coefficient distributions, edge density, and triangle count
across subgraphs, and collects the Jaccard files from step 08 into Table 16.

Source: origin/case-study-v2:masked-diff-graph/case_study/analysis_update.py.
Changes: paths as options, Agg backend (no plt.show), numbers saved to JSON/CSV.
The metrics and plots are unchanged.

Reads:  <subgraph-dir>/{N}_reddit_subgraphs_{size}.pt and ..._jaccard.json
Writes: <results-dir>/fig12_{deg_dist,clust_dist,edge_density,triangles}.png,
        <results-dir>/fig12_stats.json, <results-dir>/table16_jaccard.csv
"""

import argparse
import csv
import json
import os

import matplotlib

matplotlib.use('Agg')

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

# One base color per dataset
PALETTE = [
    '#4C9BE8',  # blue
    '#E8834C',  # orange
    '#4CE87A',  # green
    '#E84C6E',  # red/pink
    '#A04CE8',  # purple
    '#E8D94C',  # yellow
    '#4CE8D9',  # teal
]

siz = (7, 4)

DEFAULT_SETS = [
    '500_reddit_subgraphs_200.pt',
    '300_reddit_subgraphs_400.pt',
    '200_reddit_subgraphs_800.pt',
]
DEFAULT_LABELS = ['200-node', '400-node', '800-node']


def _darker(hex_color, factor=0.55):
    """Return a darkened version of a hex color."""
    r, g, b = mcolors.to_rgb(hex_color)
    return (r * factor, g * factor, b * factor)


def load_records(path):
    return torch.load(path, weights_only=False)


def degree_fn(G):
    return dict(G.degree())


# Per-dataset helpers

def _build_graph(E):
    adj = (E > 1).astype(int)
    return nx.from_numpy_array(adj)


def _collect_distribution(records, metric_fn):
    """Return a list-of-lists: inner list = metric values for one graph."""
    all_values = []
    for A, C, E in tqdm(records, desc='  evaluating', leave=False):
        G = _build_graph(E)
        all_values.append(list(metric_fn(G).values()))
    return all_values


def _collect_scalar(records, scalar_fn):
    """Return one scalar value per graph."""
    values = []
    for A, C, E in tqdm(records, desc='  evaluating', leave=False):
        G = _build_graph(E)
        values.append(scalar_fn(G))
    return values


# Multi-dataset plot helpers

def _plot_distribution_overlay(ax, all_values, color, label, bins=50):
    """Plot per-graph histogram lines + bold pooled line for one dataset."""
    flat = [v for vals in all_values for v in vals]
    for vals in all_values:
        counts, bin_edges = np.histogram(vals, bins=bins, density=True)
        ax.plot(bin_edges[:-1], counts, alpha=0.08, color=color, linewidth=0.8)
    counts, bin_edges = np.histogram(flat, bins=bins, density=True)
    ax.plot(bin_edges[:-1], counts, color=_darker(color), linewidth=2.2, label=label)


def _plot_scalar_hist(ax, values, color, label, bins=20):
    """Plot histogram + mean vline for one dataset."""
    mean_val = np.mean(values)
    ax.hist(values, bins=bins, color=color, alpha=0.35, edgecolor='none', label=label)
    ax.axvline(mean_val, color=_darker(color), linewidth=2.2,
               linestyle='--', label=f'{label} mean: {mean_val:.4f}')


# Analysis functions. Each returns {label: values} so the numbers can be saved.

def analyze_distribution(dataset_paths, dataset_labels, metric_fn, xlabel, title, save_path):
    """Overlay distribution plots for multiple datasets."""
    fig, ax = plt.subplots(figsize=siz)
    out = {}

    for idx, (path, label) in enumerate(zip(dataset_paths, dataset_labels)):
        color = PALETTE[idx % len(PALETTE)]
        print(f'[{label}] loading...')
        records = load_records(path)
        all_values = _collect_distribution(records, metric_fn)
        _plot_distribution_overlay(ax, all_values, color, label)
        out[label] = all_values

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'Saved -> {save_path}')
    return out


def _density(G):
    N = G.number_of_nodes()
    return 2 * G.number_of_edges() / (N * (N - 1)) if N > 1 else 0.0


def _triangles(G):
    return sum(nx.triangles(G).values()) // 3


def analyze_edge_density(dataset_paths, dataset_labels, save_path):
    fig, ax = plt.subplots(figsize=siz)
    out = {}

    for idx, (path, label) in enumerate(zip(dataset_paths, dataset_labels)):
        color = PALETTE[idx % len(PALETTE)]
        print(f'[{label}] loading...')
        records = load_records(path)
        densities = _collect_scalar(records, _density)
        _plot_scalar_hist(ax, densities, color, label)
        out[label] = densities

    ax.set_xlabel('Edge Density')
    ax.set_ylabel('Count')
    ax.set_title('Edge Density across Subgraphs')
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'Saved -> {save_path}')
    return out


def analyze_triangle_count(dataset_paths, dataset_labels, save_path):
    fig, ax = plt.subplots(figsize=siz)
    out = {}

    for idx, (path, label) in enumerate(zip(dataset_paths, dataset_labels)):
        color = PALETTE[idx % len(PALETTE)]
        print(f'[{label}] loading...')
        records = load_records(path)
        counts = _collect_scalar(records, _triangles)
        _plot_scalar_hist(ax, counts, color, label)
        print(f'  {label}: mean {np.mean(counts):.0f}, std {np.std(counts):.0f}')
        out[label] = counts

    ax.set_xlabel('Triangle Count')
    ax.set_ylabel('Count')
    ax.set_title('Triangle Count across Subgraphs')
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'Saved -> {save_path}')
    return out


def write_table16(dataset_paths, out_csv):
    """Collect the step 08 Jaccard files into one CSV."""
    rows = []
    for path in dataset_paths:
        jpath = path[:-len('.pt')] + '_jaccard.json'
        if not os.path.exists(jpath):
            print(f'No Jaccard file for {path}, skipped in Table 16')
            continue
        with open(jpath) as f:
            j = json.load(f)
        rows.append({
            'graph_size': j['subgraph_size'],
            'num_graphs': j['num_subgraphs'],
            'base_seed': j['base_seed'],
            'mean_jaccard': round(j['mean'], 6),
            'max_jaccard': round(j['max'], 6),
        })
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['graph_size', 'num_graphs', 'base_seed',
                                          'mean_jaccard', 'max_jaccard'])
        w.writeheader()
        w.writerows(rows)
    print(f'Saved -> {out_csv}')
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data-dir', default='../data/redditwalk')
    parser.add_argument('--subgraph-dir', default=None, help='default: <data-dir>/subgraphs')
    parser.add_argument('--results-dir', default='../results')
    parser.add_argument('--sets', nargs='+', default=DEFAULT_SETS, help='.pt file names')
    parser.add_argument('--labels', nargs='+', default=DEFAULT_LABELS)
    args = parser.parse_args(argv)
    if len(args.sets) != len(args.labels):
        parser.error('--sets and --labels need the same length')

    sub_dir = args.subgraph_dir or os.path.join(args.data_dir, 'subgraphs')
    paths = [os.path.join(sub_dir, s) for s in args.sets]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f'Missing datasets (run step 08 first): {missing}')

    res = args.results_dir
    os.makedirs(res, exist_ok=True)
    labels = args.labels

    clust = analyze_distribution(paths, labels, nx.clustering, 'Clustering Coefficient',
                                 'Clustering Coefficient Distribution across Subgraphs',
                                 os.path.join(res, 'fig12_clust_dist.png'))
    deg = analyze_distribution(paths, labels, degree_fn, 'Degree',
                               'Degree Distribution across Subgraphs',
                               os.path.join(res, 'fig12_deg_dist.png'))
    dens = analyze_edge_density(paths, labels, os.path.join(res, 'fig12_edge_density.png'))
    tri = analyze_triangle_count(paths, labels, os.path.join(res, 'fig12_triangles.png'))

    stats = {}
    for label, set_name in zip(labels, args.sets):
        flat_deg = [v for vals in deg[label] for v in vals]
        flat_clust = [v for vals in clust[label] for v in vals]
        stats[label] = {
            'dataset': set_name,
            'num_graphs': len(dens[label]),
            'edge_density_mean': float(np.mean(dens[label])),
            'edge_density_std': float(np.std(dens[label])),
            'triangles_mean': float(np.mean(tri[label])),
            'triangles_std': float(np.std(tri[label])),
            'degree_mean': float(np.mean(flat_deg)),
            'clustering_mean': float(np.mean(flat_clust)),
        }
    with open(os.path.join(res, 'fig12_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
        f.write('\n')
    print(f"Saved -> {os.path.join(res, 'fig12_stats.json')}")

    table = write_table16(paths, os.path.join(res, 'table16_jaccard.csv'))
    return stats, table


if __name__ == '__main__':
    main()

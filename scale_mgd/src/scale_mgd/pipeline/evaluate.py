"""
Evaluation steps: graph MMDs (Table 15), molecule metrics (Table 14), triangle MMD (Fig 9).

All steps share one set of generated graphs, stored in the context on first use.
"""

import json
import os

import networkx as nx
import numpy as np

from ..metrics.graph_mmd import compute_all_metrics, compute_mmd, gaussian_tv
from ..metrics.molecules import canonical_smiles_set, molecule_metrics
from .plots import plot_triangle_distribution
from .runner import PipelineStep
from .sample import dataset_to_nx, generate_graphs


def ensure_generated(context, store_raw_tensors=False):
    """Generate graphs once (settings from context['cfg'].eval) and cache them in the context."""
    cfg = context["cfg"]
    need = "gen_graphs_nx" not in context or (store_raw_tensors and "gen_raw_tensors" not in context)
    if not need:
        return
    ev = cfg.eval
    edge_slots = ev.gen_edge_slots if ev.gen_edge_slots is not None else context["M_max"]
    print(f"Generating {ev.num_samples} graphs (sampler={ev.sampler}, edge slots={edge_slots}, "
          f"temperature={ev.temperature})...")
    graphs, raw_edges, raw_tensors = generate_graphs(
        context["model"], context["dataset"], context["alphas"], context["device"],
        num_samples=ev.num_samples, batch_size=ev.batch_size, temperature=ev.temperature,
        sampler=ev.sampler, edge_slots=edge_slots, schedules=context.get("schedules"),
        store_raw_tensors=store_raw_tensors,
    )
    context["gen_graphs_nx"] = graphs
    context["gen_raw_edge_lists"] = raw_edges
    if store_raw_tensors:
        context["gen_raw_tensors"] = raw_tensors


def ref_graphs(context, split):
    """Reference graphs for a split ('train' -> context['dataset'], 'test' -> context['dataset_test'])."""
    key = f"ref_graphs_nx__{split}"
    if key not in context:
        ds = context["dataset"] if split == "train" else context.get(f"dataset_{split}")
        if ds is None:
            return None
        print(f"Building reference graphs from the {split} split...")
        context[key] = dataset_to_nx(ds, context["cfg"].eval.max_ref_graphs, context["rng"])
    return context[key]


def _jsonable(results):
    return {k: (float(v) if v == v else None) for k, v in results.items()}


class SparseDiffMetricsStep(PipelineStep):
    """Degree / Cluster / Spectre / RBF MMD against each reference split (Table 15)."""

    def __init__(self, output_dir, store_raw_tensors=False):
        super().__init__("Graph MMD metrics (Table 15)")
        self.output_dir = output_dir
        self.store_raw_tensors = store_raw_tensors

    def execute(self, context):
        ensure_generated(context, self.store_raw_tensors)
        gen = context["gen_graphs_nx"]
        os.makedirs(self.output_dir, exist_ok=True)
        for split in context["cfg"].eval.ref_splits:
            ref = ref_graphs(context, split)
            if ref is None:
                print(f"Skipping reference split '{split}' (not loaded)")
                continue
            print(f"[{split}] {len(ref)} reference / {len(gen)} generated graphs")
            results = compute_all_metrics(ref, gen, skip_rbf=context["cfg"].eval.skip_rbf,
                                          device=context["device"])
            context[f"sparsediff_metrics_{split}"] = results
            for k, v in results.items():
                print(f"  {k:<12} {v:.6f}")
            out = _jsonable(results)
            out["num_ref_graphs"] = len(ref)
            out["num_gen_graphs"] = len(gen)
            with open(os.path.join(self.output_dir, f"sparsediff_metrics_{split}.json"), "w") as f:
                json.dump(out, f, indent=2)


class MoleculeValidityStep(PipelineStep):
    """Validity / uniqueness / novelty with RDKit (Table 14). Needs store_raw_tensors=True upstream."""

    def __init__(self, output_dir):
        super().__init__("Molecule metrics (Table 14)")
        self.output_dir = output_dir

    def execute(self, context):
        ensure_generated(context, store_raw_tensors=True)
        dataset = context["dataset"]
        train_smiles = None
        if hasattr(dataset, "data") and "CAN_SMILES" in getattr(dataset.data, "columns", []):
            print("Canonicalizing training SMILES for novelty...")
            train_smiles = canonical_smiles_set(dataset.data["CAN_SMILES"].astype(str).values)
        metrics = molecule_metrics(context["gen_raw_tensors"], dataset, train_smiles)
        context["mol_metrics"] = metrics
        print(f"  validity {metrics['validity']:.4f}  uniqueness {metrics['uniqueness']:.4f}  "
              f"novelty {metrics['novelty']:.4f}  ({metrics['valid_count']}/{metrics['total']} valid)")
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "mol_metrics.json"), "w") as f:
            json.dump(_jsonable(metrics), f, indent=2)


def triangle_counts(graphs):
    return np.array(
        [sum(nx.triangles(G).values()) // 3 if G.number_of_nodes() else 0 for G in graphs],
        dtype=float,
    )


def triangle_mmd(ref_vals, gen_vals):
    """
    MMD between one 99-bin histogram of reference triangle counts and one of generated counts,
    with the Gaussian TV kernel (sigma 1). With one histogram per side this is
    2 - 2 exp(-TV^2 / 2), so it saturates at 2 - 2 e^-0.5 = 0.786939 when the histograms do not overlap.
    """
    if len(ref_vals) == 0 or len(gen_vals) == 0:
        return float("nan")
    lo = min(ref_vals.min(), gen_vals.min())
    hi = max(ref_vals.max(), gen_vals.max())
    if hi - lo < 1e-12:
        hi = lo + 1.0
    bins = np.linspace(lo, hi, 100)
    ref_hist, _ = np.histogram(ref_vals, bins=bins, density=False)
    gen_hist, _ = np.histogram(gen_vals, bins=bins, density=False)
    return compute_mmd([ref_hist], [gen_hist], kernel=gaussian_tv)


def _duplicate_edges(raw_edge_lists):
    total = 0
    for edges in raw_edge_lists:
        counts = {}
        for u, v in edges:
            key = (min(u, v), max(u, v))
            counts[key] = counts.get(key, 0) + 1
        total += sum(c - 1 for c in counts.values() if c > 1)
    return total


class TriangleEvalStep(PipelineStep):
    """Fig 9: triangle count histogram (train reference vs generated) and triangle MMD."""

    def __init__(self, output_dir, figsize=(10, 6)):
        super().__init__("Triangle distribution (Fig 9)")
        self.output_dir = output_dir
        self.figsize = figsize

    def execute(self, context):
        ensure_generated(context)
        ref = ref_graphs(context, "train")
        gen = context["gen_graphs_nx"]
        ref_tri, gen_tri = triangle_counts(ref), triangle_counts(gen)
        mmd_tri = triangle_mmd(ref_tri, gen_tri)

        os.makedirs(self.output_dir, exist_ok=True)
        plot_triangle_distribution(ref_tri, gen_tri, mmd_tri,
                                   os.path.join(self.output_dir, "triangle_distribution.png"),
                                   self.figsize)
        raw = context.get("gen_raw_edge_lists", [])
        dup = _duplicate_edges(raw)
        context["triangle_mmd"] = mmd_tri
        print(f"  triangle MMD {mmd_tri:.6f}, duplicate generated edges {dup}")
        with open(os.path.join(self.output_dir, "triangle_metrics.txt"), "w") as f:
            f.write(f"Triangle MMD: {mmd_tri}\n")
            f.write(f"Ref graphs: {len(ref)}\n")
            f.write(f"Gen graphs: {len(gen)}\n")
            f.write(f"Duplicate edges (gen, raw): {dup}\n")
            f.write(f"Avg duplicate edges/gen graph: {dup / len(raw) if raw else 0.0:.3f}\n")

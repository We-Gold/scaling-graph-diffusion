"""Report figures: edge count distribution (Fig 8) and triangle count distribution (Fig 9)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .runner import PipelineStep  # noqa: E402
from .sample import sample_edge_slots, unique_edge_counts  # noqa: E402


def save_figure(saveable, path, **kwargs):
    """Save a figure as the given path (.png) and as a .pdf next to it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    saveable.savefig(path, **kwargs)
    saveable.savefig(path.with_suffix(".pdf"), **kwargs)


def plot_edge_count_distribution(true_counts, gen_counts, save_path, figsize=(10, 6)):
    plt.figure(figsize=figsize)
    plt.hist(true_counts, bins=20, alpha=0.5, label="True Dataset", density=True, color="blue")
    plt.hist(gen_counts, bins=20, alpha=0.5, label="Generated", density=True, color="orange")
    plt.xlabel("Number of Edges")
    plt.ylabel("Density")
    plt.title("Edge Count Distribution (True vs Generated)")
    plt.legend()
    plt.tight_layout()
    save_figure(plt, save_path)
    plt.close()


def plot_triangle_distribution(ref_tri, gen_tri, mmd_tri, save_path, figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    max_tri = max(ref_tri.max() if len(ref_tri) else 0, gen_tri.max() if len(gen_tri) else 0)
    tri_bins = np.arange(0, max_tri + 2) - 0.5
    ax.hist(ref_tri, bins=tri_bins, alpha=0.5, label="Dataset", density=True, color="blue")
    ax.hist(gen_tri, bins=tri_bins, alpha=0.5, label="Generated", density=True, color="orange")
    ax.set_xlabel("Number of Triangles")
    ax.set_ylabel("Density")
    ax.set_title(f"Triangle Count Distribution\n(MMD = {mmd_tri:.6f})")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, save_path)
    plt.close(fig)


class EdgeCountPlotStep(PipelineStep):
    """Fig 8: number of edges per graph, dataset vs generated."""

    def __init__(self, save_path, num_samples, sampler="heuristic_argmax", temperature=0.5,
                 figsize=(10, 6)):
        super().__init__("Edge count distribution (Fig 8)")
        self.save_path = save_path
        self.num_samples = num_samples
        self.sampler = sampler
        self.temperature = temperature
        self.figsize = figsize

    def execute(self, context):
        dataset = context["dataset"]
        true_counts = [int((np.array(dataset[i][2]) > 1).sum() // 2) for i in range(len(dataset))]

        print(f"Generating {self.num_samples} graphs for the edge count plot "
              f"(sampler={self.sampler})...")
        batches = sample_edge_slots(
            context["model"], dataset, context["alphas"], context["device"],
            num_samples=self.num_samples, M_max=context["M_max"], sampler=self.sampler,
            temperature=self.temperature, schedules=context.get("schedules"),
        )
        gen_counts = unique_edge_counts(batches, dataset.max_length)
        plot_edge_count_distribution(true_counts, gen_counts, self.save_path, self.figsize)
        print(f"Edge count plot saved to {self.save_path}")

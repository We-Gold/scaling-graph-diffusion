"""Test script to verify the novel non-edge sampling algorithm
matches the behavior of the original sample_non_existing_edges_batched."""

import torch
import time
from torch_geometric.utils import to_dense_adj

from sparse_diffusion.diffusion.sample_edges import (
    sample_non_existing_edges_batched,
    sample_non_existing_edges_novel,
)


def verify_non_edges(edge_index_sampled, existing_edge_index, num_nodes, batch):
    """Verify that all sampled edges are truly non-existing edges."""
    # Build set of existing edges
    existing_set = set()
    for i in range(existing_edge_index.shape[1]):
        u, v = existing_edge_index[0, i].item(), existing_edge_index[1, i].item()
        existing_set.add((u, v))

    # Check all sampled edges
    for i in range(edge_index_sampled.shape[1]):
        u, v = edge_index_sampled[0, i].item(), edge_index_sampled[1, i].item()
        assert (u, v) not in existing_set, f"Sampled edge ({u}, {v}) exists!"
        assert u != v, f"Self-loop ({u}, {v})!"
        assert u < v, f"Not upper triangle: ({u}, {v})"

    # Check no duplicates
    edges = [(edge_index_sampled[0, i].item(), edge_index_sampled[1, i].item())
             for i in range(edge_index_sampled.shape[1])]
    assert len(edges) == len(set(edges)), "Duplicate sampled edges!"

    return True


def test_single_graph():
    """Test with a single graph."""
    print("=== Test: Single graph ===")
    num_nodes = torch.tensor([8], dtype=torch.long)
    batch = torch.zeros(8, dtype=torch.long)

    # Create some existing edges (upper triangle)
    existing_edge_index = torch.tensor([[0, 0, 1, 2, 3], [1, 3, 4, 5, 7]], dtype=torch.long)

    num_edges_to_sample = torch.tensor([5], dtype=torch.long)

    result = sample_non_existing_edges_novel(
        num_edges_to_sample, existing_edge_index, num_nodes, batch
    )

    print(f"  Sampled {result.shape[1]} edges")
    print(f"  Edges: {result.t().tolist()}")
    assert verify_non_edges(result, existing_edge_index, num_nodes, batch)
    print("  OK: All sampled edges are valid non-edges")


def test_batched_graphs():
    """Test with a batch of graphs."""
    print("\n=== Test: Batched graphs ===")
    num_nodes = torch.tensor([5, 7], dtype=torch.long)
    batch = torch.cat([torch.zeros(5), torch.ones(7)]).long()

    # Graph 0: edges among nodes 0-4
    existing_0 = torch.tensor([[0, 1, 2], [1, 3, 4]], dtype=torch.long)
    # Graph 1: edges among nodes 5-11
    existing_1 = torch.tensor([[5, 5, 6, 7], [6, 8, 9, 11]], dtype=torch.long)
    existing_edge_index = torch.hstack([existing_0, existing_1])

    num_edges_to_sample = torch.tensor([3, 5], dtype=torch.long)

    result = sample_non_existing_edges_novel(
        num_edges_to_sample, existing_edge_index, num_nodes, batch
    )

    print(f"  Sampled {result.shape[1]} edges (expected {num_edges_to_sample.sum().item()})")
    print(f"  Edges: {result.t().tolist()}")
    assert result.shape[1] == num_edges_to_sample.sum().item()
    assert verify_non_edges(result, existing_edge_index, num_nodes, batch)
    print("  OK: All sampled edges are valid non-edges")


def test_empty_graph():
    """Test with a graph that has no existing edges."""
    print("\n=== Test: Empty graph ===")
    num_nodes = torch.tensor([6], dtype=torch.long)
    batch = torch.zeros(6, dtype=torch.long)
    existing_edge_index = torch.zeros((2, 0), dtype=torch.long)

    num_edges_to_sample = torch.tensor([10], dtype=torch.long)

    result = sample_non_existing_edges_novel(
        num_edges_to_sample, existing_edge_index, num_nodes, batch
    )

    print(f"  Sampled {result.shape[1]} edges from empty graph")
    assert result.shape[1] == 10
    assert verify_non_edges(result, existing_edge_index, num_nodes, batch)
    print("  OK: All sampled edges are valid non-edges")


def test_consistency_with_original():
    """Test that novel and original methods produce valid results for the same input."""
    print("\n=== Test: Consistency check ===")
    torch.manual_seed(42)

    for trial in range(10):
        n = torch.randint(5, 20, (1,)).item()
        num_nodes = torch.tensor([n], dtype=torch.long)
        batch = torch.zeros(n, dtype=torch.long)

        # Create random existing edges
        num_possible = n * (n - 1) // 2
        num_existing = torch.randint(0, num_possible // 2, (1,)).item()

        rows, cols = [], []
        edges_set = set()
        while len(edges_set) < num_existing:
            u = torch.randint(0, n, (1,)).item()
            v = torch.randint(0, n, (1,)).item()
            if u < v and (u, v) not in edges_set:
                edges_set.add((u, v))
                rows.append(u)
                cols.append(v)

        if num_existing > 0:
            existing_edge_index = torch.tensor([rows, cols], dtype=torch.long)
        else:
            existing_edge_index = torch.zeros((2, 0), dtype=torch.long)

        num_non_existing = num_possible - num_existing
        to_sample = min(num_non_existing, max(1, num_non_existing // 3))
        num_edges_to_sample = torch.tensor([to_sample], dtype=torch.long)

        result_novel = sample_non_existing_edges_novel(
            num_edges_to_sample, existing_edge_index, num_nodes, batch
        )
        result_original = sample_non_existing_edges_batched(
            num_edges_to_sample, existing_edge_index, num_nodes, batch
        )

        assert result_novel.shape[1] == to_sample, f"Trial {trial}: wrong count"
        assert result_original.shape[1] == to_sample, f"Trial {trial}: wrong count (original)"
        assert verify_non_edges(result_novel, existing_edge_index, num_nodes, batch)
        assert verify_non_edges(result_original, existing_edge_index, num_nodes, batch)

    print(f"  OK: All {10} random trials passed for both methods")


def benchmark_performance():
    # Not a test: timing printout only. Run with `python tests/test_novel_sampling.py`.
    """Compare performance of novel vs original."""
    print("\n=== Performance comparison ===")

    sizes = [50, 100, 500, 1000]
    for n in sizes:
        num_nodes = torch.tensor([n], dtype=torch.long)
        batch = torch.zeros(n, dtype=torch.long)

        # Create ~20% existing edges
        num_possible = n * (n - 1) // 2
        num_existing = num_possible // 5
        perm = torch.randperm(num_possible)[:num_existing]

        # Convert condensed to matrix
        from sparse_diffusion.diffusion.sample_edges_utils import condensed_to_matrix_index
        existing_edge_index = condensed_to_matrix_index(perm, n)

        to_sample = min(num_possible - num_existing, num_existing // 2)
        num_edges_to_sample = torch.tensor([to_sample], dtype=torch.long)

        # Time novel
        t0 = time.time()
        for _ in range(5):
            sample_non_existing_edges_novel(
                num_edges_to_sample, existing_edge_index, num_nodes, batch
            )
        t_novel = (time.time() - t0) / 5

        # Time original
        t0 = time.time()
        for _ in range(5):
            sample_non_existing_edges_batched(
                num_edges_to_sample, existing_edge_index, num_nodes, batch
            )
        t_original = (time.time() - t0) / 5

        speedup = t_original / t_novel if t_novel > 0 else float('inf')
        print(f"  n={n:5d}: novel={t_novel:.4f}s, original={t_original:.4f}s, speedup={speedup:.2f}x")


if __name__ == "__main__":
    test_single_graph()
    test_batched_graphs()
    test_empty_graph()
    test_consistency_with_original()
    benchmark_performance()
    print("\nOK: All tests passed!")

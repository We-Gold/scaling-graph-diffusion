# Edge sampling benchmark (SparserDiff Experiment 1)

Pure Python benchmark of three ways to sample k missing edges from a sparse graph: materialize the
adjacency matrix, rejection sampling, and our space-efficient sampler (prefix array, Floyd sampling,
binary search). It measures time and peak memory as the graph grows.

## Report anchors

| Anchor | Code or file |
| --- | --- |
| Sec 4.2, Algorithm 4, Fig 4 | `novel_sampling_faster_floyd`, `sample_without_replacement_floyd` in `src/edge_sampling/algorithms.py` |
| Sec 5.2.1, Table 9 (design) | `src/edge_sampling/benchmark.py` (sizes, fill = min(floor(3.8 n), n(n-1)/2), k = 1000) |
| Sec 5.2.2, Fig 6a, 6b (all 3 methods) | `results/fig06a_edge_sampling_memory_all.*`, `results/fig06b_edge_sampling_time_all.*` |
| Sec 5.2.2, Fig 6c, 6d (Rejection vs Ours) | `results/fig06c_edge_sampling_memory_rej_vs_ours.*`, `results/fig06d_edge_sampling_time_rej_vs_ours.*` |
| Fig 6 raw data | `results/fig06_edge_sampling.csv` |

Method names in code, CSV, and report:

| Function | CSV and fig 6 label | Report name |
| --- | --- | --- |
| `materialize_adj_matrix_fast` | SparseDiff | Materialize Adjacency Matrix |
| `rejection_sampling` | Rejection Sampling | Rejection Sampling |
| `novel_sampling_faster_floyd` | Ours | Ours |

`materialize_adj_matrix` is not a benchmarked method. It builds the random input graph for each size and seed.

## Setup

```bash
cd benchmarks/edge_sampling
uv sync
```

Python 3.12, numpy and matplotlib only. CPU only.

## Run

Re-plot fig 6 from the committed CSV (a few seconds):

```bash
uv run python -m edge_sampling.benchmark --from-saved --results-file ../../results/fig06_edge_sampling.csv \
    --output-prefix fig06ab_all
uv run python -m edge_sampling.benchmark --from-saved --results-file ../../results/fig06_edge_sampling.csv \
    --include-methods "Rejection Sampling,Ours" --output-prefix fig06cd_rej_vs_ours
```

Each prefix writes `<prefix>_memory_scaling.{png,pdf}` and `<prefix>_speed_scaling.{png,pdf}`.

Full rerun (table 9 sizes, seeds 42, 62, 123, 456, 789). Writes `sampling_scaling_results.csv` in the current folder:

```bash
uv run python -m edge_sampling.benchmark --output-prefix fig06ab_all
```

Smoke run: `uv run python -m edge_sampling.benchmark --sizes 400,800 --seeds 42 --results-file /tmp/es.csv --output-prefix /tmp/es`.

Turing/SLURM: not needed. The benchmark is CPU only and runs on a laptop. The full rerun needs several GB of RAM,
because the input generator builds all n(n-1)/2 pairs as Python tuples (32 million at n = 8000).

## Data

None. Input graphs are generated at random from the seed.

## Results

`results/fig06_edge_sampling.csv` (150 rows: 10 sizes x 3 methods x 5 seeds; columns
`num_nodes,method,seed,time_seconds,peak_memory_kb`) and the four fig 6 panels as PNG and PDF. These are the
original files from the MQP run (2026-03-06), copied unchanged. Re-plotting the CSV gives the same curves; only
font rendering may differ.

## Missing

Experiment 2 (Sec 5.2, tables 10-11, fig 7) is missing: no script or results were found in the source repo. It
used the PyTorch sampler inside SparseDiff, so it would need the `sparserdiff/` environment. See `sparserdiff/README.md`.

## Notes

- Table 9 lists seed 42, but the CSV and fig 6 (error bars are std over seeds) use 5 seeds. Pass `--seeds 42` for one seed.
- Fig 6 labels the materialize baseline "SparseDiff". It is the report's "Materialize Adjacency Matrix" baseline,
  not SparseDiff's own sampler.
- Time is `time.perf_counter`. Memory is the `tracemalloc` peak in KB, which counts only Python heap allocations.
  Numbers depend on the machine; use them for relative comparison.
- This version uses Floyd sampling for the random indices, as in Sec 4.2. The PyTorch version in `sparserdiff/`
  uses torch rejection sampling for the indices instead.
- The constant `percent_filled = 2.8` is a fill multiplier, not a percentage: the input graph has (1 + 2.8) n edges.
- Source: branch `sparse-diff-new-alg` at `205829a`, folder `sparse-diff-new-alg-demo/`.

# Scaling benchmark (report 5.1)

Time, peak GPU memory, and RAM of five graph diffusion models (Scale-MGD, SparserDiff, DiGress, SparseDiff,
MG-Diff) on synthetic graphs of growing size. Four stages: forward diffusion, one model pass, one training step,
and full sampling. One profiler per model, all writing the same CSV format, merged by `report.py`.

## Report anchors

| Anchor | Code or file |
| --- | --- |
| 5.1.1, Table 1 (network sizes) | `configs.py` (`N_LAYERS`, `HIDDEN_DIMS`, `HIDDEN_MLP_DIMS`); Scale-MGD sizes in `profile_scale_mgd.py` |
| 5.1.1, Table 2 (graph types, batch 128) | `configs.py` (`CONFIGS`) |
| 5.1.1 protocol (4 stages, 10 seeds) | `common.py` |
| Tables 3-8, Scale-MGD rows | `profile_scale_mgd.py`, log `results/tables03-08_scale_mgd_profile.txt` |
| Tables 3-8, DiGress rows | `profile_digress.py` + vendored `digress/`, log `results/tables03-08_digress_profile.txt` |
| Tables 3-8, MG-Diff rows | `profile_mg_diff.py` (uses `baselines/mg_diff`), log `results/tables03-08_mg_diff_profile.txt` |
| Tables 3-8, SparseDiff and SparserDiff rows | `profile_sparsediff.py --sampler original` / `--sampler novel` (uses `sparserdiff/`). No log, see Missing |
| Tables 3-8 layout | `report.py` |
| Fig 5 (GPU MB vs N) | Missing, see below |

## Setup

Two environments. Scale-MGD and MG-Diff run in the root uv workspace (Python 3.12, torch 2.8). DiGress, SparseDiff,
and SparserDiff run in the `sparserdiff/` uv project (Python 3.9, torch 2.0.1, PyG 2.3.1, same stack that
upstream DiGress pins).

```bash
cd benchmarks/scaling
uv sync                                   # workspace member: scale-mgd, mg-diff, psutil
(cd ../../sparserdiff && uv sync)         # see sparserdiff/README.md
```

All scripts are plain modules in this folder, run with `python -m` from here. `common.py`, `configs.py`,
`report.py`, and the sparserdiff-env profilers use Python 3.9 syntax.

## Run

Local smoke test (CPU, about 2 min in total): Molecular Graph only, batch 4, 1 seed, 2 steps.

```bash
uv run python -m profile_scale_mgd --smoke
uv run python -m profile_mg_diff --smoke
uv run --project ../../sparserdiff python -m profile_digress --smoke
uv run --project ../../sparserdiff python -m profile_sparsediff --smoke --sampler original   # SparseDiff
uv run --project ../../sparserdiff python -m profile_sparsediff --smoke --sampler novel      # SparserDiff
uv run python -m report --raw out/raw --out out/tables
uv run pytest -q && uv run --project ../../sparserdiff python -m pytest -q tests
```

Each profiler writes `out/raw/<model>.csv` (one row per graph type, seed, and stage) and prints the old text log
format. `report.py` writes `out/tables/tables03-08_summary.csv` and `out/tables/tableNN.md` (rows in report order).
Options: `--configs molecular,large` (keys in `configs.py`), `--seeds 3` or `--seeds 10,42`, `--steps`,
`--batch-size`, `--out`; `profile_sparsediff.py` also has `--edge-fraction` (default 0.1).

The copied logs can be turned into the same tables:

```bash
uv run python -m report --out out/logs --log \
    Scale-MGD=../../results/tables03-08_scale_mgd_profile.txt \
    DiGress=../../results/tables03-08_digress_profile.txt \
    MG-Diff=../../results/tables03-08_mg_diff_profile.txt
```

Turing/SLURM (report: 1x A100-80G, 4 CPUs, 32 GB). One job per model, so RAM is per process as in the original runs:

```bash
cd benchmarks/scaling && mkdir -p logs
for m in scale_mgd mg_diff digress sparsediff sparserdiff; do sbatch slurm/profile.sh $m; done
# after all jobs finish
uv run python -m report --raw out/raw --out out/tables
```

The script asks for `--gres=gpu:1`, since the Turing GPU type name is not recorded. A comment in the script shows
where to add it (for example `gpu:a100:1`; check with `sinfo -o "%P %G"`). The longest part is SparseDiff and
SparserDiff sampling at N = 200 (about 235 s per seed).

## Data

None. Inputs are random tensors generated from the seed.

## Results

| File | Backs | Match with the report |
| --- | --- | --- |
| `results/tables03-08_scale_mgd_profile.txt` | Scale-MGD rows of tables 3-8 | Exact (72 values), except Table 3 forward diffusion time: log 0.0076±0.0216, report 0.0004±0.0001. The report dropped seed 10, a 0.07 s warmup outlier |
| `results/tables03-08_digress_profile.txt` | DiGress rows | Table 6 exact. Tables 3, 4, 5, 7, 8 within about 1% (earlier run than the report) |
| `results/tables03-08_mg_diff_profile.txt` | MG-Diff rows | GPU column exact in all tables, Table 6 exact. Time and RAM in tables 3, 4, 5, 7, 8 from an earlier run |

These are the original logs from `origin/sparse-diff-new-algo:SparseDiff_repo/` (Turing, 2026-02-22), copied
unchanged. Column order in the logs is Time, RAM, GPU (the report uses Time, GPU, RAM).
`tests/test_report.py` checks report cells against them. The Table 6 training step is `nan` for DiGress and MG-Diff
(and in the report for all models except Scale-MGD): CUDA out of memory on the 80 GB GPU.

## Missing

| Item | Report | Who likely has it |
| --- | --- | --- |
| SparseDiff and SparserDiff logs, and the profiler version that made them | SparseDiff and SparserDiff rows of tables 3-8 | Hien Pham, Turing. The only committed SparseDiff profiler used edge fraction 0.5 and the novel sampler hard-coded; its logs do not match the report (not copied). `profile_sparsediff.py` is our rebuild (edge fraction 0.1 from the report text, sampler flag), so a rerun will not match the published rows exactly |
| Final DiGress and MG-Diff logs | DiGress and MG-Diff rows of tables 3, 4, 5, 7, 8 (later rerun) | Hien Pham, Turing |
| Fig 5 plot script | Fig 5 | Not in any branch or stash. Not reconstructed |
| Environment of the original runs | all | Not recorded. Scale-MGD ran in a Python 3.12 / torch 2.8 uv project |

## Notes

- Diffusion steps differ by model. The Scale-MGD rows used 20 reverse steps (default of its profiler, stated in
  its log). DiGress, MG-Diff, and SparseDiff used 10, as the report text says. So the Scale-MGD sampling (backward
  process) times in tables 3-8 are for 2x the steps of the other models. The profilers keep these defaults
  (`profile_scale_mgd.py` 20, the others 10) to reproduce the tables. Use `--steps 10` for an equal-steps run.
- Edge fraction for SparseDiff and SparserDiff is 0.1 (report 5.1.1). The old profiler used 0.5.
- SparserDiff differs from SparseDiff only in the forward noise (new-edge sampler, `model.use_novel_sampling`).
  The forward diffusion and training step stages use it. Sampling uses upstream code for both.
- Inputs are not Erdos-Renyi G(N, M) as the report says. DiGress and MG-Diff use random dense type tensors,
  SparseDiff uses 2N random edges, and Scale-MGD uses min(N^2, 5N) random (u, v, type) rows. Kept so numbers stay
  comparable. `n_edges` in the CSV is the Table 2 value M.
- Stage bodies are copied from the original profilers. DiGress, SparseDiff, and Scale-MGD time with `time.time()`
  and no `cuda.synchronize()`; MG-Diff synchronizes. The training step uses a dummy loss (mean of outputs) for
  DiGress and SparseDiff, cross-entropy for Scale-MGD, and the real VLB loss plus an AdamW step for MG-Diff.
  Scale-MGD sampling is greedy argmax decoding.
- Stage names: Noise Addition = Forward Diffusion, Backward Process = Sampling (report names in `report.py`).
- Scale-MGD is the base `SparseGNNMaskedDiffusionModel` (no query edges), hidden 256, 3 GNN layers, 5 iterations,
  4 decoder layers. The current model has `max_nodes + 1` pointer heads (the profiled version had `max_nodes`).
- Small changes to the original scripts: shared seed loop and printer, Table 1 sizes from `configs.py`, a failed
  seed is recorded as NaN instead of stopping the graph type, Scale-MGD now prints stage errors, and the literal
  `\n` in error messages is a newline.
- `np.random.default_rng(seed)` in `set_all_seeds` does not seed numpy's global RNG. Kept from the originals.
- MG-Diff uses a dummy charge stream with one class, which gives a numpy divide warning. See
  `baselines/mg_diff/README.md`.
- `digress/` is a 14-file subset of cvignac/DiGress@bdfa10c (MIT). See `digress/UPSTREAM.md` for the patches.
- Source: `origin/sparse-diff-new-algo` (f4b7365), files `SparseDiff_repo/src/profile_diffusion.py`,
  `SparseDiff/profile_sparsediff.py`, `mg_diff/profile_mgdiff.py`, and
  `wpi-graph-ai-mqp-25-26/masked-diff-graph/profile_md4_sparse.py`.

# SparserDiff

Vendored SparseDiff ([qym7/SparseDiff@d9c4b20](https://github.com/qym7/SparseDiff/tree/d9c4b20cf6e6e2c59913165c1d7eaaf70a9174de), v1.1.0, MIT, see `LICENSE`) plus our SparserDiff patch: a new-edge sampler that does not build the dense candidate set. The flag `model.use_novel_sampling` switches it on (default `False` = upstream SparseDiff). The upstream README is kept as `README.upstream.md` (its conda and `requirements.txt` steps are replaced by uv here).

Git history: commit 1 is the pristine upstream copy, commit 2 is the code patch, commit 3 adds this README, the SLURM script, and the uv project. Use `git diff <commit1> <commit2> -- sparserdiff` to see the patch.

## Report anchors

| Anchor | What | Code |
| --- | --- | --- |
| 3.2 | SparseDiff background | upstream, unchanged |
| 4.2, Alg. 4, Fig 4 | SparserDiff new-edge sampler | `sample_non_existing_edges_novel` in `sparse_diffusion/diffusion/sample_edges.py` |
| 5.2 Exp 3, Tables 12-13 (and SparserDiff rows of Tables 14-15) | SparseDiff vs SparserDiff on QM9 (20 epochs) and Ego (100 epochs) | `slurm/train.sh`, flag in `configs/model/discrete.yaml` |
| 5.1, Tables 3-8 | SparseDiff/SparserDiff profiling | profiler lives in `benchmarks/scaling/`, runs in this env |

## Setup

This is its own uv project (Python 3.9, torch 2.0.1, PyG 2.3.1, dgl 1.1.2). It is not a member of the root uv workspace, since the workspace uses Python 3.12. The lock covers macOS arm64 (CPU) and Linux x86_64 (torch 2.0.1+cu118).

```bash
cd sparserdiff
uv sync
(cd sparse_diffusion/analysis/orca && g++ -O2 -std=c++11 -o orca orca.cpp)   # once per machine, for orbit MMD
uv run pytest -q tests
```

On macOS, `pyemd` and `torch-geometric` build from source (needs Xcode command line tools). graph-tool is not installed; it is only used by SBM metrics. For SBM, use upstream's conda recipe (`conda install -c conda-forge graph-tool=2.45`).

## Run (local)

The Hydra entry point must run from `sparse_diffusion/` (data and outputs resolve to `sparserdiff/data/` and `sparserdiff/outputs/`).

```bash
cd sparse_diffusion
uv run python main.py +experiment=debug dataset=ego model.use_novel_sampling=true general.wandb=disabled
```

Training (flag off = SparseDiff, flag on = SparserDiff). Overrides follow the report text (sec 5.2), since the original configs are missing:

```bash
# Table 12, QM9 without H. model/discrete.yaml defaults give 5 layers, 500-step cosine schedule, edge_fraction 0.5.
uv run python main.py dataset=qm9 dataset.remove_h=True train.n_epochs=20 train.batch_size=32 \
    general.name=t12_qm9_sparserdiff model.use_novel_sampling=true general.wandb=offline
# Table 13, Ego. experiment/ego.yaml gives 8 layers, 1000 steps, edge_fraction 0.1, batch 32.
uv run python main.py +experiment=ego train.n_epochs=100 \
    general.name=t13_ego_sparserdiff model.use_novel_sampling=true general.wandb=offline
```

Use `model.use_novel_sampling=false` and `..._sparsediff` names for the baseline. Do not put `debug` or `test` in `general.name` (reserved upstream). Training ends with `trainer.test`, which samples `general.final_model_samples_to_generate` graphs (QM9 10000, Ego 151) and prints the metrics. To evaluate a checkpoint, add `general.test_only=/abs/path/to/last.ckpt`.

## Run (Turing/SLURM)

```bash
cd sparserdiff && mkdir -p logs
sbatch slurm/train.sh qm9 true    # or: qm9 false, ego true, ego false
```

`uv` must be on PATH. The first `uv sync` downloads the cu118 torch wheel (about 2 GB). Compile ORCA on Turing too. The Turing GPU type name for `--gres` is unknown (report: 1x L40S on `short`), so the script asks for `gpu:1`. torch 2.0.1+cu118 has sm_86 kernels, which run on L40S. Run the Ego debug command once with `general.gpus=1` to confirm.

## Data

Downloaded automatically on first run by upstream code. Not in git.

| Dataset | Used by | Source | Path | Size |
| --- | --- | --- | --- | --- |
| QM9 (no H) | Table 12 | deepchemdata S3 `qm9.zip` + figshare uncharacterized list | `sparserdiff/data/qm9/` | 364 MB |
| Ego | Table 13 | `tufts-ml/graph-generation-EDGE` `graphs/Ego.pkl` (757 graphs) | `sparserdiff/data/ego/` | 126 MB |

QM9 split: shuffle with `random_state=42`, 100,000 train, 10% test, rest val. Ego split (upstream quirk, kept): seed 1234, 606 train, 151 test, and val is the first 151 train graphs (val overlaps train).

## Results

No result files exist. Tables 12, 13, and 17 exist only as numbers in the report. After a rerun, suggested names: `results/table12_qm9_sparsediff_vs_sparserdiff.csv`, `results/table13_ego_sparsediff_vs_sparserdiff.csv`.

## Missing

Not in any branch of the old repo. Not reconstructed here.

| Item | Report | What it needs | Who likely has it |
| --- | --- | --- | --- |
| Exp 2 sampler benchmark script and results | Tables 10-11, Fig 7 | Script that loads the QM9 and Ego test sets through the SparseDiff datamodules, sweeps batch sizes (QM9 16-512, Ego 4-64), times `sample_non_existing_edges_batched` vs `_novel` (3 warmup + 10 timed rounds, median), reports analytic peak memory, plus CSVs and the Fig 7 plot. The rule for `num_edges_to_sample` is not stated in the report. | Hien Pham (wrote the PyTorch sampler; Turing `/home/hpham/wpi-graph-ai-mqp-25-26`) |
| Exp 3 configs, SLURM script, logs, checkpoints, wandb runs | Tables 12-13 (and rows in 14-15) | Exact overrides (unknown: `model.use_charge`, `lambda_train`, seed, number of generated samples), the code version with the flag (never pushed; the branch hard-codes the novel sampler), `outputs/` with `last.ckpt` and metric files. | Hien Pham (likely; L40S on `short`), Turing home dir or wandb |
| SparserDiff on RedditWalk: dataset adapter, configs, checkpoints, metrics, plot scripts | 6.4, Table 17, Figs 13-14 | A dataset module for the RedditWalk `.pt` files (built by `redditwalk/`), hooks in `main.py` and `metrics/sampling_metrics.py`, configs (1000 epochs, batch 16/8/4, edge_fraction 0.5/0.5/0.04, 32 generated graphs), checkpoints, and plot scripts for Figs 13-14. | Unknown. Likely Hien (runs) with David Dechantsreiter's `.pt` datasets (`masked-diff-graph/data/NovelReddit/` or Turing) |
| SparserDiff Planar row | Table 15 | Any config or log. No run is described in the report. | Unknown |

## Notes

- Report sec 4.2 describes Floyd sampling. This PyTorch sampler uses torch rejection sampling for the random indices (the Floyd version is in `benchmarks/edge_sampling/`). The docstring says so.
- Only the forward noise path has the flag. The reverse (sampling) path uses upstream `sampled_condensed_indices_uniformly`, as in the original branch.
- The flag is our re-implementation of the toggle used for Tables 12-13 (that version was never pushed). The sampler body is unchanged, but runs will not match the report bit for bit.
- QM9 settings not in the report (`use_charge`, `lambda_train`, number of samples) use upstream defaults.
- Ego with 100 epochs: `check_val_every_n_epochs` is 1000 in `experiment/ego.yaml`, so no validation runs. `last.ckpt` and the final test still run.
- Changes to upstream beyond the sampler: package-style imports in the model (so `tests/` and `benchmarks/scaling` can import it from any directory; `main.py` still uses bare imports), FCD falls back to CPU without CUDA, `nx.from_numpy_array` for networkx 2.8+, support for `model.extra_features=null`. `setup.py` and `requirements.txt` are replaced by `pyproject.toml`.
- Omitted from the vendored copy: `sparse_diffusion/metrics/fcd/ChemNet_v0.13_pretrained.pt` (5.6 MB, unused; FCD uses the `fcd_torch` package).
- wandb 0.15.4 resolves with a new sentry-sdk (deprecation warnings only). If online logging breaks, add `constraint-dependencies = ["sentry-sdk<2"]` under `[tool.uv]`.
- `psutil` is a dependency for the `benchmarks/scaling` profilers, which run with `uv run --project ../../sparserdiff`.

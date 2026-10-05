# Vendored DiGress subset

- Upstream: https://github.com/cvignac/DiGress
- Commit: `bdfa10cdedc9e274c755ac17dc42d633ea8f5909` (2026-09-10). These 14 files have not changed upstream since
  `df02c01` (2024-11-01).
- License: MIT, `LICENSE` (copied unchanged).
- Upstream path: `src/`. Only the import closure of `src/diffusion_model_discrete.py` is kept (14 files).
  Datasets, analysis, molecular metrics, `main.py`, and configs are left out.

Used only by `../profile_digress.py` (report 5.1, DiGress rows of tables 3-8). It runs in the `sparserdiff/`
environment (Python 3.9, torch 2.0.1, PyG 2.3.1, pytorch_lightning 2.0.4, torchmetrics 0.11.4, wandb 0.15.4),
the same stack that upstream DiGress pins.

## Local patches

The first git commit that adds this folder is the pristine upstream copy. The patch is the next commit.

1. Package-style imports. Upstream mixes `from src...` with `from models...`, `from diffusion...`,
   `from metrics...` (which only work with `src/` on `sys.path`). All of these became `from digress...`.
2. `models/transformer_model.py`: `GraphTransformer` passes `dim_ffy=hidden_dims.get('dim_ffy', 2048)` to
   `XEyTransformerLayer`. Upstream ignores `dim_ffy` and always uses 2048. Needed for the report's Table 1
   (dff_y = 64). The old MQP copy (`origin/sparse-diff-new-algo:SparseDiff_repo/src/`) had the same patch.

Nothing else differs from upstream. The old MQP copy also removed one `print` in the `marginal` transition branch;
the profiler uses the `uniform` transition, so that change is not needed and not applied.

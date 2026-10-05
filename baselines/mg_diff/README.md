# MG-Diff (reimplementation)

Our reimplementation of MG-Diff (Zhang et al., PLOS ONE 2025, doi 10.1371/journal.pone.0331450).
It is a small library used only by the scaling benchmark in `benchmarks/scaling/`.
Nothing is trained or evaluated here.

## Report anchors

| Anchor | What |
| --- | --- |
| 3.1 | MG-Diff background: mask-and-replace noise (`scheduler.py`) |
| 5.1, footnote 1, Table 1 | MG-Diff network used in the profiling (`model.py`, Table 1 sizes via `hidden_dims` / `hidden_mlp_dims`) |
| Tables 3-8, Fig 5 | MG-Diff rows, produced by `benchmarks/scaling` with this package |
| Table 14 | Not backed by this code. See Notes |

## Setup

From the repo root: `uv sync` (workspace member `mg-diff`, Python 3.12, torch 2.8, numpy).

## Run

Local smoke test (CPU, about 30 s):

```bash
cd baselines/mg_diff
uv run pytest -q
```

Profiling (local and Turing/SLURM): see `benchmarks/scaling/README.md`.

API:

```python
from mg_diff import D2GraphTransformer, DiffusionTransformer

model = D2GraphTransformer(atom_dim=9, charge_dim=1, edge_dim=4, num_layers=5, d_model=256,
                           num_heads=8, dff=256, position=True, dropout=0.0,
                           hidden_dims={...}, hidden_mlp_dims={...})
sched = DiffusionTransformer(mask_id=0, a_classes=9, c_classes=1, e_classes=4, model=model,
                             diffusion_step=10, max_length=50)
loss = sched(A, C, E)[-1]          # A (b,n), C (b,n), E (b,n,n) int tensors, E symmetric
[(A, C, E)] = sched.sample(128)    # one list entry per internal batch of 256
```

## Data

None.

## Results

None of its own. MG-Diff profiling results are listed in `benchmarks/scaling/README.md`.

## Missing

Nothing needed for the report. MG-Diff ZINC-250K training logs and checkpoints were not migrated (see Notes).

## Notes

- Files:
  - `scheduler.py`: mask-and-replace scheduler (`DiffusionTransformer`), one stream each for atoms, charges, and edges. MASK is class 0.
    Adapted from microsoft/VQ-Diffusion (MIT, `LICENSE-VQ-Diffusion`).
  - `model.py`: `D2GraphTransformer`, written by us from the MG-Diff paper supplement: discrete embeddings,
    sinusoidal positions, PAD mask (`pad_idx = 1`), output heads without the MASK class.
  - `transformer.py`, `layers.py`: the DiGress graph transformer backbone (MIT, `LICENSE-DiGress`).
    So report footnote 1 ("from scratch") applies to the MG-Diff wrapper, not to the backbone.
- Source: `origin/sparse-diff-new-algo:SparseDiff_repo/mg_diff/` in the old repo (the profiled copy).
  Changes are imports, a local `PlaceHolder`, and removed dead code. Same parameters, loss, and samples
  as the old code under a fixed seed.
- Table 14: the MG-Diff row (0.964 / 0.994 / 0.985) and the LatentGAN, JT-VAE, VAE, MolGPT, and DiGress rows
  are quoted from MG-Diff paper Table 1 (ZINC-250K). We did not reproduce them.
- The old ZINC training and evaluation scripts were dropped. They had a bug: `zinc250k_train.py` trained
  `DenoisingNetwork` (`new_model.py`), while `zinc250k_evaluation.py` built `D2GraphTransformer`, so the
  saved checkpoints could not load in the evaluation script. Not fixed, since the code is not kept.
- The profiler uses `charge_dim = c_classes = 1` (dummy charge). This divides by zero in `alpha_schedule`
  (numpy RuntimeWarning, NaN buffers for the charge stream). The loss stays finite and the behavior is kept,
  since the profiling only measures time and memory.
- `DiffusionTransformer.device` refers to `model.to_logits`, which `D2GraphTransformer` does not have
  (left over from VQ-Diffusion). It is never called. Kept as-is.

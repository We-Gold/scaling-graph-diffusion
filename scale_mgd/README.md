# Scale-MGD

Sparse masked graph diffusion with a random-unmasking GNN encoder and an O(M) edge decoder.
Ported from the MQP repo, branch `masked-diff-framework-comparison` at a3a980b (`masked-diff-graph/`).

## Report anchors

| Report | Code |
| --- | --- |
| 4.1.1 sparse representation (node list + M edge slots, PAD, MASK, virtual node) | `data/collate.py` |
| 4.1.2 Eq. 29 schedule, atomic edge masking | `diffusion/discrete.py`, `diffusion/sparse_graph.py` |
| 4.1.3 Eq. 33-36 VLB + adaptive auxiliary loss | `diffusion/sparse_graph.py` (`SparseGraphDiffusionLoss`) |
| 4.1.4 Alg. 1, Fig. 3 network | `models/encoder.py`, `models/gnn.py` (`SparseGNNMaskedDiffusionModel`) |
| Query-edge variant (used for the 5.3 runs) | `models/query_edges.py` (`SparseGNNQueryEdgesModel`) |
| 4.1.5 Alg. 2 training | `pipeline/train.py` |
| 4.1.6 Alg. 3 sampling | `diffusion/sparse_graph.py` (`sample_posterior_sparse_graph`), `pipeline/sample.py` |
| 5.3 Table 14 (ZINC250k validity / uniqueness / novelty) | `metrics/molecules.py`, `scripts/run.py` |
| 5.3 Table 15 (Ego, Planar MMDs; Reference rows) | `metrics/graph_mmd.py`, `scripts/run.py`, `scripts/reference_mmd.py` |
| 5.3 Fig. 8 (edge counts), Fig. 9 (triangles) | `pipeline/plots.py`, `scripts/regen_plots.py` |
| 5.1 Tables 3-8 (dependency only) | `scale_mgd.SparseGNNMaskedDiffusionModel` is imported by `benchmarks/scaling/` |

Paths in the table are under `src/scale_mgd/`.

## Setup

From the repo root: `uv sync`. Python 3.12, torch 2.8, PyG 2.6.1, numpy < 2.

## Run

Local, from `scale_mgd/`:

```bash
# Train + evaluate one dataset with the default config. Output: outputs/ce_query_edges/<dataset>_<time>/
uv run python scripts/run.py --config configs/ce_query_edges.yaml --dataset planar
# VLB config (report text, Alg. 2-3)
uv run python scripts/run.py --config configs/vlb_query_edges.yaml --dataset planar
# All datasets of a config in sequence; override single values with --set
uv run python scripts/run.py --config configs/ce_query_edges.yaml --dataset all --set train.epochs=5
# Table 15 "Reference" rows (train vs test MMD)
uv run python scripts/reference_mmd.py --dataset ego
# Figs 8-9 in report style (5x3, 64 samples) from a finished run -> <run>/viz/regen/
uv run python scripts/regen_plots.py --run-dir outputs/ce_query_edges/planar_<time>
# Figs 8-9 from an old MQP checkpoint (module names are unchanged, so it loads strictly)
uv run python scripts/regen_plots.py --config configs/ce_query_edges.yaml --dataset ego \
  --checkpoint /path/to/gnn_query_edges__ce__ego/checkpoints/model_final.pt --out-dir figs/ego
# Tests (tiny CPU train + sample + eval + regen, no download); add -m network for a Planar run
uv run pytest -q
```

Turing (SLURM), from `scale_mgd/`. One dataset per job. The GPU type is a commented option in the scripts.

```bash
sbatch --export=ALL,CONFIG=ce_query_edges,DATASET=zinc250k slurm/run.sh
sbatch --export=ALL,CONFIG=ce_query_edges,DATASET=ego slurm/run.sh      # batch 4, slowest
sbatch --export=ALL,RUN_DIR=outputs/ce_query_edges/ego_<time> slurm/regen_plots.sh
```

Where the numbers land: `<run>/result.json` (and a one-row `results.csv` with the old column names).
Table 14: `mol_validity`, `mol_uniqueness`, `mol_novelty`. Table 15: `degree_mmd`, `cluster_mmd`,
`spectre_mmd`, `rbf_mmd` (train reference) and the same keys with `_test` (test reference).
Fig 8: `viz/regen/edge_count_dist.{png,pdf}`. Fig 9: `viz/regen/triangle_distribution.{png,pdf}`, MMD in `triangle_metrics.txt`.

## Configs

| | `ce_query_edges.yaml` (default) | `vlb_query_edges.yaml` |
| --- | --- | --- |
| Purpose | Reproduces the 2026-03-03 runs behind Figs 8-9 and Tables 14-15 | Follows the report text (4.1.2-4.1.6) |
| Noise | Bernoulli masking, keep-probability `linspace(0.9999, 0.0001, 100)` | Eq. 29 absorbing + uniform schedule, atomic edges |
| Loss | Plain CE on x0 | VLB with x0 marginalization + adaptive aux CE (weight 1e-4) |
| Sampler (Table 15, Fig 9) | Predict x0 (temperature 0.5), re-mask with prob. 1 - alpha_{t-1} | Alg. 3 posterior sampling |
| Sampler (Fig 8) | Argmax, unmask-only heuristic (as in the original) | Alg. 3 posterior sampling |
| Edge slots at generation | 100 (hardcoded in the original, kept) | Dataset M_max |

Both use the query-edge network (edge fraction 0.3, hidden 128, 4 MP layers, 5 unmasking trials,
4 decoder layers, dropout 0.2), 20 epochs, AdamW lr 1e-3, T = 100. Set `model.name: gnn` for the
plain Alg. 1 network. So the report numbers come from a setting that differs from the report text:
Bernoulli masking with linear alphas and a predict-then-remask sampler, not Eq. 29 and Alg. 3.
The VLB config (from a3a980b) was never run at full scale.

## Data

All datasets live under the repo-root `data/` (override with `SGD_DATA_ROOT`).

| Dataset | Path | How to get it |
| --- | --- | --- |
| ZINC250k | `data/zinc250k/raw.csv` | Manual. Kaggle `250k_rndm_zinc_drugs_clean_3.csv`, renamed to `raw.csv`. First run writes the 80/10/10 split and `processed_*.pt` (about 2.3 GB for train). |
| Planar | `data/planar/planar_64_200.pt` | Auto-download from the SPECTRE repo (6.5 MB) |
| Ego | `data/ego/Ego.pkl` | Auto-download from the EDGE repo (7.7 MB) |

SLURM nodes need internet for Planar and Ego, or copy the files there first.

## Results

Nothing is committed yet. The report runs are on Turing (see Missing). Earlier local runs are not
copied on purpose, so they are not confused with report numbers. Once the Turing files are fetched,
they go to `results/table14_scale_mgd_zinc250k.json`, `results/table15_scale_mgd_{ego,planar}.json`,
`results/table15_reference_{ego,planar}.json`, `results/fig08{a,b,c}_edge_count_{zinc250k,planar,ego}.png`,
`results/fig09{a,b,c}_triangles_{zinc250k,planar,ego}.png`.

## Missing

| What | Probably where | Needed for |
| --- | --- | --- |
| `outputs/experiment_architecture_comparison_2026-03-03_10-14-07/gnn_query_edges__ce__zinc250k/` (result.json, eval/, viz/regen/, checkpoints/model_final.pt) | Turing, Weaver's clone of the MQP repo under `masked-diff-graph/` | Table 14 Scale-MGD row, Figs 8a, 9a |
| `outputs/experiment_architecture_comparison_2026-03-03_11-03-11/gnn_query_edges__ce__{ego,planar}/` plus `results.csv`, `optimal_baseline_*.json` | Turing (same) | Table 15 Scale-MGD and Reference rows, Figs 8b-c, 9b-c |
| A finished Ego run | No local Ego run finished; only the Turing run above | Table 15 Ego rows, Figs 8c, 9c |

The old checkpoints load with `scripts/regen_plots.py --checkpoint ...`.

## Notes

Known issues (kept as in the original, so the report runs can be reproduced):

- Fig 9 triangle MMD is saturated. It compares one histogram with one histogram (TV kernel), so it equals
  2 - 2 exp(-TV^2 / 2). The Planar value 0.786939 = 2 - 2 e^-0.5 is the maximum (no overlap).
- Ego edge truncation: training keeps only the first M_max = 500 edges of each graph. 194 of the 757 Ego
  graphs have more and are cut. The report says m = max edges in the dataset. Training now prints a warning.
- Table 15 / Fig 9 samples have at most 100 edge slots in the CE config (hardcoded in the original),
  even for Planar (M_max 200) and Ego (M_max 500). Fig 8 uses the dataset M_max.
- Edge-type marginals are passed to the encoder in training only. At sampling the encoder uses uniform
  types. pi_u and pi_v are never computed (uniform). The marginal counts double-count real edges (kept).
- The original runs set no seed. `seed` (default 0) is new, so numbers will not match bit for bit.
- Table 15 reference column: unknown whether the Scale-MGD rows used the train or the test reference.
  The code reports both. The Planar Reference row mixes two loaders: degree and cluster match the
  SPECTRE loader, spectre and RBF match the default loader (the only one ported). The Ego Reference RBF
  (0.009) matches no local run.
- ZINC bonds: `reconstruct_molecule` compares RDKit bond types with strings, so every generated bond is
  added as SINGLE. Table 14 validity is 0.00 in the report.
- Query edges use `torch.randperm(n(n-1)/2)`, which is O(n^2) memory. Only the base model is profiled in 5.1.

Changes from the original: one config and one dataset per call; failures exit non-zero; checkpoint
loading is strict; novelty compares against canonicalized training SMILES (the original compared to
raw strings); reference subsampling uses a seeded RNG; ablation options, dead code, and dev plots are removed.

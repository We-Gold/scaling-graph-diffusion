# Results

Small files that back report tables and figures, named by report ID.

| File | Report | Made by | Notes |
| --- | --- | --- | --- |
| `tables03-08_{scale_mgd,digress,mg_diff}_profile.txt` | 5.1, tables 3-8 | `benchmarks/scaling` | Original logs (Turing, 2026-02-22). Scale-MGD matches tables 3-8 exactly, except the table 3 forward diffusion time (the report dropped seed 10). DiGress matches table 6 exactly. MG-Diff matches table 6 and the whole GPU column; its other cells come from a later run that was not saved. No SparseDiff or SparserDiff logs |
| `fig06_edge_sampling.csv` | 5.2 Exp 1, fig 6 | `benchmarks/edge_sampling` | Raw data: 10 sizes x 3 methods x 5 seeds |
| `fig06{a,b}_edge_sampling_{memory,time}_all.{png,pdf}` | fig 6a, 6b | `benchmarks/edge_sampling` | All 3 methods |
| `fig06{c,d}_edge_sampling_{memory,time}_rej_vs_ours.{png,pdf}` | fig 6c, 6d | `benchmarks/edge_sampling` | Rejection sampling vs ours |
| `sec6_1_wcc_stats.json` | 6.1 | `redditwalk/scripts/01_giant_component.py` | Matches the report exactly |

Not produced yet (inputs or runs are missing, see the root README):

- `table14_scale_mgd_zinc250k.json`, `table15_scale_mgd_{ego,planar}.json`, `table15_reference_{ego,planar}.json`,
  `fig08*`, `fig09*`: from `scale_mgd`, after the 2026-03-03 Turing runs are fetched.
- `table12_*`, `table13_*`: from `sparserdiff`, after a rerun.
- `fig12_*`, `fig12_stats.json`, `table16_jaccard.csv`: from `redditwalk/scripts/09_dataset_stats.py`.

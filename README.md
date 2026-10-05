# Scaling Diffusion Models to Large Sparse Graphs

Code for the WPI Major Qualifying Project of the same name (March 2026), by David Dechantsreiter,
Weaver Goldman, Botao Hu, and Hien Pham, advised by Fabricio Murai and Oren Mangoubi.
The report is [`MQP_GenAI_for_Graph_25_26_5.pdf`](MQP_GenAI_for_Graph_25_26_5.pdf).

The project has two methods and one dataset:

- **Scale-MGD**: masked discrete diffusion on a sparse edge list with a GNN, linear memory in nodes plus edges.
- **SparserDiff**: SparseDiff with a new-edge sampler that never builds the full set of non-existent edges.
- **RedditWalk**: a large-scale benchmark built from the Reddit Hyperlink Network.

## Report to code

| Report | Folder | Environment | Runs on |
| --- | --- | --- | --- |
| 4.1, 5.3 (tables 14-15, figs 8-9) | [`scale_mgd/`](scale_mgd/README.md) | uv workspace | local, Turing |
| 4.2, 5.2 Exp 2-3 (tables 10-13, fig 7), 6.4 | [`sparserdiff/`](sparserdiff/README.md) | own uv project (Python 3.9) | local (CPU smoke), Turing |
| 5.2 Exp 1 (table 9, fig 6) | [`benchmarks/edge_sampling/`](benchmarks/edge_sampling/README.md) | uv workspace | local |
| 5.1 (tables 1-8, fig 5) | [`benchmarks/scaling/`](benchmarks/scaling/README.md) | uv workspace + `sparserdiff/` env | local (CPU smoke), Turing |
| 3.1, used in 5.1 | [`baselines/mg_diff/`](baselines/mg_diff/README.md) | uv workspace | local |
| 6.1-6.3 (figs 10-12, table 16) | [`redditwalk/`](redditwalk/README.md) | uv workspace | local |
| none (teaching visuals) | [`viz/`](viz/README.md) | uv workspace + Next.js | local |
| | [`results/`](results/README.md) | | small files that back report tables and figures |
| | [`data/`](data/README.md) | | datasets (not in git) |

## Setup

Needs [uv](https://docs.astral.sh/uv/). For some parts: Node 20+ (`viz/web`), a C++ compiler (`sparserdiff` orbit
metrics), and ffmpeg (`viz/animations`).

```bash
uv sync --all-packages             # workspace: Python 3.12, torch 2.8, PyG 2.6.1
(cd sparserdiff && uv sync)        # SparseDiff stack: Python 3.9, torch 2.0.1, PyG 2.3.1
```

## Smoke tests

All run on a laptop CPU in a few minutes, with no downloads unless noted.

```bash
(cd scale_mgd && uv run pytest -q)
(cd sparserdiff && uv run pytest -q tests)
(cd baselines/mg_diff && uv run pytest -q)
(cd benchmarks/edge_sampling && uv run python -m edge_sampling.benchmark --sizes 400 --seeds 42 --results-file /tmp/es.csv --output-prefix /tmp/es)
(cd benchmarks/scaling && uv run python -m profile_scale_mgd --smoke)   # see its README for the other models
(cd redditwalk && uv run pytest -q)
```

Full runs, data, and Turing (SLURM) jobs are in each folder's README.

## Turing (SLURM)

SLURM scripts are in `<part>/slurm/` and use relative paths and `uv run`. They request `--gres=gpu:1`. The
GPU type names on Turing are not recorded here; run `sinfo -o "%P %G"` on Turing and add the type
(for example `gpu:a100:1`) where the script comments show. The report used an A100-80G for 5.1 and an L40S
on the `short` partition for 5.2 Exp 3.

Note from Weaver: I haven't tested these Turing/SLURM scripts, but they should capture the right ideas at least.

## Missing

Some code and results were never committed to the original repo. They are probably on Turing or with the
person listed. Each part README has the details.

| What | Report | Likely with |
| --- | --- | --- |
| Exp 2 script (edge sampling on Ego and QM9) | tables 10-11, fig 7 | Hien Pham |
| Exp 3 configs and logs (SparseDiff vs SparserDiff) | tables 12-13 | Hien Pham |
| SparserDiff on RedditWalk: adapter, configs, checkpoints | 6.4, table 17, figs 13-14 | Hien Pham, David Dechantsreiter |
| SparserDiff Planar run | table 15 | unknown |
| Scale-MGD runs of 2026-03-03, including Ego | tables 14-15, figs 8-9 | Turing (Weaver Goldman) |
| SparseDiff and SparserDiff profiling logs, final DiGress and MG-Diff reruns, fig 5 script | 5.1 | Hien Pham |
| BERT embeddings, topic clustering, Node2Vec, `giant_component.csv` | 6.2, figs 10-11 | Hien Pham |
| RedditWalk seeds for the 200 and 400 node sets, and the `.pt` datasets | 6.3, table 16, fig 12 | David Dechantsreiter |

## Source

Rebuilt from the original project repo (`wpi-graph-ai-mqp-25-26`, about 37 branches). Only code that backs the
report, or that kept code depends on, was kept. Each part README names its source branch and commit.

# RedditWalk

Builds the RedditWalk dataset from the SNAP Reddit Hyperlinks network and computes its statistics.

## Report anchors

| Section | Content | Code |
| --- | --- | --- |
| 6.1 | GWCC of the merged body + title graph (67,180 nodes, 712 WCCs, GWCC 65,648 = 97.72%) | `scripts/01_giant_component.py`, `scripts/02_filter_edges.py` |
| 6.2 step 1 | Subreddit descriptions from `about.json` (46,668 = 71.09% with description) | `scripts/03_fetch_descriptions.py`, `scripts/04_retry_rate_limited.py` |
| 6.2 steps 2-4, Figs 10-11 | BERT embeddings, 20 topic clusters, PCA, Node2Vec + t-SNE | missing (steps 05-07) |
| 6.3, Table 16 | Random walk with restart (p = 0.3), 500x200, 300x400, 200x800, mean pairwise Jaccard | `scripts/08_sample_subgraphs.py`, `src/redditwalk/dataset.py` |
| 6.3, Fig 12 | Degree, clustering, edge density, triangle count | `scripts/09_dataset_stats.py` |
| 6.4, Table 17, Figs 13-14 | SparserDiff on RedditWalk | missing, see `../sparserdiff/` |

## Pipeline

All data lives in `data/redditwalk/` (gitignored). Numbers 05-07 are reserved for the missing steps.

```
SNAP soc-redditHyperlinks-{body,title}.tsv
  -> 01_giant_component      giant_component_nodes.txt, wcc_stats.json, results/sec6_1_wcc_stats.json
  -> 02_filter_edges         giant_component_{body,title,combined}.csv
  -> 03_fetch_descriptions   subreddit_descriptions.tsv           (live Reddit API)
  -> 04_retry_rate_limited   subreddit_descriptions_retry.tsv     (live Reddit API)
  -> 05_merge_retry_results  [MISSING] merged descriptions TSV
  -> 06_embed_cluster        [MISSING] BERT, 20 clusters, fig 10, giant_component.csv
  -> 07_node2vec_tsne        [MISSING] fig 11 (side branch)
  -> 08_sample_subgraphs     subgraphs/{N}_reddit_subgraphs_{size}.pt, _jaccard.json, _nodes.json
  -> 09_dataset_stats        results/fig12_*.png, fig12_stats.json, table16_jaccard.csv
  -> sparserdiff/ adapter    [MISSING, 6.4]
```

Where the chain breaks:

1. After step 04: the merge script and the saved Feb 2026 descriptions TSV are missing. A new crawl gives different numbers.
2. Steps 05-07 have no code, so `giant_component.csv` cannot be rebuilt. Step 08 needs it.
3. After step 08: the three report `.pt` files are missing, and only one `base_seed` is known.
4. 6.4: no SparserDiff RedditWalk adapter, configs, checkpoints, or results.

Today steps 01-04 run on real data. Steps 08-09 run on real data once `giant_component.csv` is provided, and on synthetic data in the smoke test.

## Setup

```
uv sync            # from the repo root or from redditwalk/
```

## Run (local)

Run from `redditwalk/`. Defaults: `--data-dir ../data/redditwalk`, `--results-dir ../results`.

```
mkdir -p ../data/redditwalk && cd ../data/redditwalk
curl -O https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv
curl -O https://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv
cd -

# 6.1 (about 1 minute)
uv run scripts/01_giant_component.py
uv run scripts/02_filter_edges.py

# 6.2 step 1 (historical: live API, hours, results differ from Feb 2026)
uv run scripts/03_fetch_descriptions.py      # --resume-from <index> after an interruption
uv run scripts/04_retry_rate_limited.py

# 05-07 missing. Put giant_component.csv into ../data/redditwalk/.

# 6.3 (needs giant_component.csv)
uv run scripts/08_sample_subgraphs.py --num-subgraphs 500 --size 200 --base-seed <unknown>
uv run scripts/08_sample_subgraphs.py --num-subgraphs 300 --size 400 --base-seed <unknown>
uv run scripts/08_sample_subgraphs.py --num-subgraphs 200 --size 800 --base-seed 8
uv run scripts/09_dataset_stats.py

# smoke test (synthetic data, no network, a few seconds)
uv run pytest tests/
```

## Run (Turing/SLURM)

Not needed. Nothing here uses a GPU. Run step 03 on a machine that allows long HTTP jobs, not a compute node.

## Data

| File in `data/redditwalk/` | Source | Status |
| --- | --- | --- |
| `soc-redditHyperlinks-{body,title}.tsv` | SNAP (Kumar et al. 2018), https://snap.stanford.edu/data/soc-RedditHyperlinks.html, 319 MB + 369 MB | download |
| `giant_component_nodes.txt` | step 01, 65,648 names, SHA-256 `74ab07d51108475ebf328eea749e10e61ea3e1701ef162f575ecb524aa1b952a` (same as the file on `origin/hiepham`) | regenerate |
| `giant_component_{body,title,combined}.csv` | step 02 (286,252 + 571,271 = 857,523 rows) | regenerate |
| `subreddit_descriptions*.tsv` | steps 03-05, Feb 2026 crawl | missing (Hien) |
| `giant_component.csv` | step 06 | missing (Hien) |
| `subgraphs/500_reddit_subgraphs_200.pt`, `300_..._400.pt`, `200_..._800.pt` | step 08 | missing (David) |

`giant_component.csv` columns, as step 08 reads them: all SNAP columns plus `source_type`, `target_type`
(cluster id 0-19, or -1 for no description) and `source_topic`, `target_topic` (label).

## Results

| File | Report | Status |
| --- | --- | --- |
| `results/sec6_1_wcc_stats.json` | 6.1 | committed, matches the report exactly |
| `results/fig12_{deg_dist,clust_dist,edge_density,triangles}.png`, `results/fig12_stats.json` | Fig 12 | not yet: needs the real `.pt` sets. Compare: triangles 14,019 / 66,408 / 256,473, density 0.105 / 0.079 / 0.055 |
| `results/table16_jaccard.csv` | Table 16 | not yet. Compare: 0.0461 / 0.0714 / 0.1102 |

Do not commit results from synthetic data.

## Missing

| Item | Needed for | Owner / likely place |
| --- | --- | --- |
| `merge_retry_results.py` (step 05) | 6.2 step 1 numbers | Hien Pham, Turing `reddit_hyperlinks_dataset/` |
| Saved descriptions TSVs (original, retry, merged) | 6.2 numbers (46,668 = 71.09%) | Hien, Turing |
| BERT `bert-base-uncased` [CLS] embedding code and embeddings | 6.2 step 2 | Hien |
| 20-topic clustering, label assignment, PCA plot | 6.2 step 3, Fig 10 | Hien |
| Join step and `giant_component.csv` | input of step 08 | Hien |
| Node2Vec + t-SNE code | 6.2 step 4, Fig 11 | Hien |
| `base_seed` for the 500x200 and 300x400 sets (800 set used 8) | Table 16, Fig 12 | David Dechantsreiter |
| The three `.pt` datasets | Fig 12, Table 16, 6.4 | David (old `masked-diff-graph/data/NovelReddit/`, or Turing) |
| SparserDiff RedditWalk adapter, configs, checkpoints, metrics | 6.4, Table 17, Figs 13-14 | unknown, see `../sparserdiff/README.md` |

The `.pt` files store no node names, so Table 16 cannot be recomputed from them. Only a re-run of
step 08 with the right seed and the same CSV row order reproduces it. Step 08 now also saves
`_jaccard.json` and `_nodes.json`.

## Notes

- Sources: `origin/hiepham:reddit_hyperlinks_dataset/` (steps 01-04) and
  `origin/case-study-v2:masked-diff-graph/case_study/{reddit_dataset.py,analysis_update.py}` (steps 08-09).
- Header fix: the original step 01 read the SNAP header line as an edge (67,182 nodes, 713 WCCs).
  It now uses `header=0` and reproduces 6.1 exactly (67,180 nodes, 712 WCCs, GWCC 65,648 = 97.72%).
  The GWCC node set is the same both ways.
- Walk termination check: the walk loops forever if the start node's component is smaller than the
  target size. The sampler now raises `ValueError` in that case. It draws no random numbers, so
  successful runs are unchanged.
- Known deviations from the report text (code kept as-is):
  - Walks run only on nodes with a description (about 46,668), not the full 65,648-node GWCC.
    This graph can be disconnected (hence the check above).
  - "Stratified" topic seeding is one uniformly random topic per subgraph, then a random node of
    that topic. Balanced in expectation, not by quota.
- The directed multigraph becomes an undirected `nx.Graph`. For repeated pairs the last row's
  `LINK_SENTIMENT` wins. Edge types: 0 MASK, 1 NO_BOND, 2 Positive, 3 Negative. Node type = cluster id + 2.
- Split: 80/20 with `torch.Generator().manual_seed(0)` (test sizes 100 / 60 / 40, as in 6.4).
- The 20 topic labels in `RedditDataset.types` are the only record of the cluster names. They differ
  from the two examples in the report text.
- `.pt` files made before commit `ec760aa` (2026-03-03) of the old repo ignore `subgraph_size` and are not report datasets.

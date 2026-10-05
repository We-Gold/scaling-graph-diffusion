"""Synthetic stand-ins for the RedditWalk inputs (smoke test only, not report data)."""

import numpy as np
import networkx as nx
import pandas as pd

SNAP_HEADER = "SOURCE_SUBREDDIT\tTARGET_SUBREDDIT\tPOST_ID\tTIMESTAMP\tLINK_SENTIMENT\tPROPERTIES\n"


def write_giant_component_csv(path, n=600, seed=0):
    """Fake `giant_component.csv`: SNAP columns plus source/target type and topic.

    Power-law cluster graph, random topic 0..19 per node, 10% of nodes without a
    description (type -1), random edge direction, sentiment +1 / -1.
    """
    rng = np.random.default_rng(seed)
    G = nx.powerlaw_cluster_graph(n, 4, 0.3, seed=seed)
    names = [f"sub{i}" for i in G.nodes()]
    topic = {i: int(rng.integers(0, 20)) for i in G.nodes()}
    for i in rng.choice(n, n // 10, replace=False):
        topic[int(i)] = -1
    rows = []
    for u, v in G.edges():
        if rng.random() < 0.5:
            u, v = v, u
        rows.append(dict(SOURCE_SUBREDDIT=names[u], TARGET_SUBREDDIT=names[v], POST_ID="x",
                         TIMESTAMP="2016-01-01 00:00:00",
                         LINK_SENTIMENT=int(rng.choice([1, -1], p=[.9, .1])), PROPERTIES="0,0",
                         source_type=topic[u], target_type=topic[v],
                         source_topic="t", target_topic="t"))
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_snap_tsvs(data_dir):
    """Two tiny SNAP-style TSVs WITH the header line.

    Components: {a, b, c, d} (giant, 4 nodes) and {e, f}. 6 nodes, 2 WCCs.
    The (a, b) link appears in both files, so dedup leaves 4 directed edges.
    """
    body = [("a", "b", 1), ("b", "c", -1), ("e", "f", 1)]
    title = [("a", "b", 1), ("c", "d", 1)]
    for name, rows in [("soc-redditHyperlinks-body.tsv", body), ("soc-redditHyperlinks-title.tsv", title)]:
        with open(f"{data_dir}/{name}", "w") as f:
            f.write(SNAP_HEADER)
            for i, (s, t, sent) in enumerate(rows):
                f.write(f"{s}\t{t}\tp{i}\t2016-01-01 00:00:00\t{sent}\t\"0.1,0.2\"\n")

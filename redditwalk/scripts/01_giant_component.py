#!/usr/bin/env python3
"""Step 01 (report 6.1): weakly connected components of the merged body + title graph.

Load both SNAP TSVs -> build directed graph -> find all WCCs -> save the giant WCC node list.

Source: origin/hiepham:reddit_hyperlinks_dataset/analyze_wccs.py.
Fix: `header=0` in both read_csv calls. The original read the SNAP header line as an edge
(67,182 nodes, 713 WCCs). With the fix the output matches 6.1 (67,180 nodes, 712 WCCs).

Writes: <data-dir>/giant_component_nodes.txt, <data-dir>/wcc_stats.json,
        <results-dir>/sec6_1_wcc_stats.json (same content).
"""

import argparse
import json
import logging
import os
from collections import Counter

import networkx as nx
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BODY_TSV = "soc-redditHyperlinks-body.tsv"
TITLE_TSV = "soc-redditHyperlinks-title.tsv"
COLUMNS = ['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 'POST_ID', 'TIMESTAMP', 'LINK_SENTIMENT', 'PROPERTIES']


def load_graph_from_both_tsvs(data_dir):
    """Load BOTH body and title TSV files and merge into single graph."""
    logger.info("Loading BOTH TSV files (body + title)")

    logger.info(f"1. Loading body TSV: {BODY_TSV}")
    df_body = pd.read_csv(
        os.path.join(data_dir, BODY_TSV),
        sep='\t',
        header=0,
        names=COLUMNS,
        usecols=['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT'],  # Only need these columns
    )
    logger.info(f"   Body edges: {len(df_body):,}")

    logger.info(f"2. Loading title TSV: {TITLE_TSV}")
    df_title = pd.read_csv(
        os.path.join(data_dir, TITLE_TSV),
        sep='\t',
        header=0,
        names=COLUMNS,
        usecols=['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT'],
    )
    logger.info(f"   Title edges: {len(df_title):,}")

    logger.info("3. Merging both datasets...")
    df_combined = pd.concat([df_body, df_title], ignore_index=True)
    n_rows = len(df_combined)
    logger.info(f"   Combined edges: {n_rows:,}")

    # Remove duplicates (same edge in both files)
    df_combined = df_combined.drop_duplicates(subset=['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT'])
    logger.info(f"   After deduplication: {len(df_combined):,}")
    logger.info(f"   Unique source subreddits: {df_combined['SOURCE_SUBREDDIT'].nunique():,}")
    logger.info(f"   Unique target subreddits: {df_combined['TARGET_SUBREDDIT'].nunique():,}")

    logger.info("4. Building graph from edge list...")
    G = nx.DiGraph()
    edges = list(zip(df_combined['SOURCE_SUBREDDIT'], df_combined['TARGET_SUBREDDIT']))
    G.add_edges_from(edges)

    logger.info("Graph created:")
    logger.info(f"  Nodes: {G.number_of_nodes():,}")
    logger.info(f"  Edges: {G.number_of_edges():,}")
    logger.info(f"  Density: {nx.density(G):.6f}")

    return G, n_rows


def analyze_wccs(G, data_dir):
    """Find and analyze all Weakly Connected Components."""
    logger.info("=" * 60)
    logger.info("ANALYZING WEAKLY CONNECTED COMPONENTS")
    logger.info("=" * 60)

    G_undirected = G.to_undirected()
    wccs = list(nx.connected_components(G_undirected))

    logger.info(f"Total WCCs found: {len(wccs):,}")

    wcc_sizes = sorted((len(wcc) for wcc in wccs), reverse=True)

    logger.info("WCC Size Statistics:")
    logger.info(f"  Minimum size: {min(wcc_sizes):,}")
    logger.info(f"  Maximum size: {max(wcc_sizes):,}")
    logger.info(f"  Mean size: {sum(wcc_sizes)/len(wcc_sizes):.1f}")
    logger.info(f"  Median size: {wcc_sizes[len(wcc_sizes)//2]:,}")

    giant_size = wcc_sizes[0]
    total_nodes = G.number_of_nodes()
    logger.info("Giant Component:")
    logger.info(f"  Size: {giant_size:,} nodes")
    logger.info(f"  Percentage: {giant_size/total_nodes*100:.2f}%")

    logger.info("Top 10 Largest WCCs:")
    for i, size in enumerate(wcc_sizes[:10], 1):
        logger.info(f"  #{i}: {size:,} nodes ({size/total_nodes*100:.2f}%)")

    size_distribution = Counter(wcc_sizes)
    logger.info("Size Distribution (nodes -> count):")
    for size in sorted(size_distribution.keys())[:20]:
        logger.info(f"  Size {size}: {size_distribution[size]:,} WCCs")
    if len(size_distribution) > 20:
        logger.info(f"  ... and {len(size_distribution) - 20} more size categories")

    isolated_count = size_distribution.get(1, 0)
    logger.info(f"Isolated nodes (size 1): {isolated_count:,} WCCs")

    giant_component = max(wccs, key=len)
    giant_nodes = sorted(giant_component)

    output_file = os.path.join(data_dir, "giant_component_nodes.txt")
    with open(output_file, 'w') as f:
        for node in giant_nodes:
            f.write(f"{node}\n")

    logger.info(f"Giant Component nodes saved to: {output_file}")
    logger.info(f"First 20 nodes: {giant_nodes[:20]}")

    return wccs, wcc_sizes, giant_component


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data-dir', default='../data/redditwalk')
    parser.add_argument('--results-dir', default='../results')
    args = parser.parse_args(argv)

    logger.info("=" * 60)
    logger.info("REDDIT HYPERLINKS WCC ANALYSIS (BODY + TITLE MERGED)")
    logger.info("=" * 60)

    G, n_rows = load_graph_from_both_tsvs(args.data_dir)
    wccs, wcc_sizes, giant_component = analyze_wccs(G, args.data_dir)

    total_nodes = G.number_of_nodes()
    stats = {
        'tsv_rows': n_rows,
        'directed_edges_dedup': G.number_of_edges(),
        'nodes': total_nodes,
        'num_wccs': len(wccs),
        'gwcc_nodes': wcc_sizes[0],
        'gwcc_percent': round(100 * wcc_sizes[0] / total_nodes, 2),
        'second_wcc_nodes': wcc_sizes[1] if len(wcc_sizes) > 1 else 0,
        'nodes_outside_gwcc': total_nodes - wcc_sizes[0],
    }
    logger.info(f"Stats: {stats}")
    with open(os.path.join(args.data_dir, 'wcc_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, 'sec6_1_wcc_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
        f.write('\n')

    logger.info("=" * 60)
    logger.info("ANALYSIS COMPLETE!")
    logger.info("=" * 60)
    return stats


if __name__ == "__main__":
    main()

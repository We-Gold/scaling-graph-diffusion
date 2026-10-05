#!/usr/bin/env python3
"""Step 02 (report 6.1): keep only links where both ends are in the giant WCC.

Keeps all original SNAP columns. The combined file is the concat of body and title links and
is NOT deduplicated (multi-edges kept).

Source: origin/hiepham:reddit_hyperlinks_dataset/filter_giant_component.py (paths only changed).

Reads:  <data-dir>/giant_component_nodes.txt, <data-dir>/soc-redditHyperlinks-{body,title}.tsv
Writes: <data-dir>/giant_component_{body,title,combined}.csv
"""

import argparse
import logging
import os

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

GIANT_COMPONENT_FILE = "giant_component_nodes.txt"
BODY_TSV = "soc-redditHyperlinks-body.tsv"
TITLE_TSV = "soc-redditHyperlinks-title.tsv"

OUTPUT_BODY = "giant_component_body.csv"
OUTPUT_TITLE = "giant_component_title.csv"
OUTPUT_COMBINED = "giant_component_combined.csv"


def load_giant_component_nodes(path):
    """Load the set of nodes in giant component."""
    logger.info(f"Loading giant component nodes from {path}")
    with open(path, 'r') as f:
        nodes = set(line.strip().lower() for line in f if line.strip())
    logger.info(f"Loaded {len(nodes):,} nodes in giant component")
    return nodes


def filter_data(input_file, output_file, giant_nodes):
    """Filter TSV to only include edges within giant component."""
    logger.info(f"Reading {input_file}")

    df = pd.read_csv(input_file, sep='\t')
    original_count = len(df)
    logger.info(f"  Original records: {original_count:,}")

    df['source_lower'] = df['SOURCE_SUBREDDIT'].str.lower()
    df['target_lower'] = df['TARGET_SUBREDDIT'].str.lower()

    # Filter: both source AND target must be in giant component
    mask = (df['source_lower'].isin(giant_nodes)) & (df['target_lower'].isin(giant_nodes))
    df_filtered = df[mask].drop(columns=['source_lower', 'target_lower'])

    filtered_count = len(df_filtered)
    removed_count = original_count - filtered_count
    logger.info(f"  Filtered records: {filtered_count:,}")
    logger.info(f"  Removed: {removed_count:,} ({100*removed_count/original_count:.1f}%)")

    df_filtered.to_csv(output_file, index=False)
    logger.info(f"  Saved to {output_file}")

    return df_filtered


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data-dir', default='../data/redditwalk')
    args = parser.parse_args(argv)
    d = args.data_dir

    logger.info("=" * 60)
    logger.info("Filter Reddit Hyperlinks to Giant Component")
    logger.info("=" * 60)

    giant_nodes = load_giant_component_nodes(os.path.join(d, GIANT_COMPONENT_FILE))

    df_body = filter_data(os.path.join(d, BODY_TSV), os.path.join(d, OUTPUT_BODY), giant_nodes)
    df_title = filter_data(os.path.join(d, TITLE_TSV), os.path.join(d, OUTPUT_TITLE), giant_nodes)

    logger.info("Creating combined file")
    df_combined = pd.concat([df_body, df_title], ignore_index=True)
    df_combined.to_csv(os.path.join(d, OUTPUT_COMBINED), index=False)

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info(f"Giant component nodes: {len(giant_nodes):,}")
    logger.info(f"Body links (filtered): {len(df_body):,}")
    logger.info(f"Title links (filtered): {len(df_title):,}")
    logger.info(f"Total links (filtered): {len(df_combined):,}")
    logger.info("=" * 60)
    return len(df_body), len(df_title), len(df_combined)


if __name__ == "__main__":
    main()

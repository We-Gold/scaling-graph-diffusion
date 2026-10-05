#!/usr/bin/env python3
"""Step 04 (report 6.2 step 1): retry the subreddits that step 03 marked rate_limited.

Historical step (live API, see step 03). Rows keep the INDEX of the original TSV.
The original merge script (05_merge_retry_results.py) is missing; see the README.

Source: origin/hiepham:reddit_hyperlinks_dataset/retry_rate_limited.py (paths only changed).

Reads:  <data-dir>/subreddit_descriptions.tsv
Writes: <data-dir>/subreddit_descriptions_retry.tsv, <data-dir>/retry_rate_limited.log
"""

import argparse
import logging
import os
import time
from collections import defaultdict

import requests

logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (compatible; SubredditDescriptionFetcher/1.0)"
DELAY_BETWEEN_REQUESTS = 1.0  # Slower for retry
INPUT_FILE = "subreddit_descriptions.tsv"
OUTPUT_FILE = "subreddit_descriptions_retry.tsv"
LOG_FILE = "retry_rate_limited.log"


def setup_logging(log_path):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        force=True,
    )


def fetch_subreddit_description(subreddit_name):
    """Fetch description using public JSON API."""
    url = f"https://www.reddit.com/r/{subreddit_name}/about.json"
    headers = {'User-Agent': USER_AGENT}

    try:
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 404:
            return 'not_found', None
        elif response.status_code == 403:
            return 'forbidden', None
        elif response.status_code == 429:
            return 'rate_limited', None
        elif response.status_code == 503:
            return 'error_503', None
        elif response.status_code != 200:
            return f'error_{response.status_code}', None

        data = response.json()

        if 'data' in data:
            desc = data['data'].get('public_description', '')
            if not desc:
                desc = data['data'].get('description', '')

            if desc and desc.strip():
                return 'success_with_desc', desc.strip()
            else:
                return 'success_no_desc', None
        else:
            return 'error_no_data', None

    except Exception as e:
        logger.warning(f"Error for {subreddit_name}: {e}")
        return 'error_unknown', None


def load_rate_limited_subreddits(input_file):
    """Load subreddits that were rate limited from TSV file."""
    rate_limited = []

    logger.info(f"Reading {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        f.readline()  # Skip header

        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                index, subreddit, status = parts[0], parts[1], parts[2]
                if status == 'rate_limited':
                    rate_limited.append((index, subreddit))

    logger.info(f"Found {len(rate_limited)} rate_limited subreddits")
    return rate_limited


def retry_rate_limited(input_file, output_file, delay=DELAY_BETWEEN_REQUESTS):
    """Retry fetching rate limited subreddits."""
    rate_limited = load_rate_limited_subreddits(input_file)

    if not rate_limited:
        logger.info("No rate_limited entries found!")
        return {}

    stats = defaultdict(int)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("INDEX\tSUBREDDIT\tSTATUS\tDESCRIPTION\n")

        for idx, (original_idx, subreddit) in enumerate(rate_limited, 1):
            status, description = fetch_subreddit_description(subreddit)
            stats[status] += 1

            desc_clean = (description or '').replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
            f.write(f"{original_idx}\t{subreddit}\t{status}\t{desc_clean}\n")
            f.flush()

            if idx % 100 == 0:
                logger.info(f"Progress: {idx}/{len(rate_limited)} ({100*idx/len(rate_limited):.1f}%)")
                logger.info(f"Stats: {dict(stats)}")

            time.sleep(delay)

    logger.info(f"Completed! Retried {len(rate_limited)} subreddits")
    logger.info(f"Final stats: {dict(stats)}")
    logger.info(f"Results saved to {output_file}")
    logger.info("Next step (05_merge_retry_results.py) is missing; see README.")
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data-dir', default='../data/redditwalk')
    args = parser.parse_args(argv)

    setup_logging(os.path.join(args.data_dir, LOG_FILE))
    logger.info("=" * 60)
    logger.info("Retry Rate Limited Subreddits")
    logger.info("=" * 60)
    return retry_rate_limited(os.path.join(args.data_dir, INPUT_FILE),
                              os.path.join(args.data_dir, OUTPUT_FILE))


if __name__ == "__main__":
    main()

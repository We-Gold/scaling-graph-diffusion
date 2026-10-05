#!/usr/bin/env python3
"""Step 03 (report 6.2 step 1): fetch subreddit descriptions from Reddit's public JSON API.

Historical step. It hits the live, rate-limited API (no auth). A new crawl gives different
numbers than the Feb 2026 crawl behind the report (subreddits banned or changed since).

Source: origin/hiepham:reddit_hyperlinks_dataset/fetch_subreddit_descriptions.py.
Changes: paths as options, dead code removed, `--resume-from` exposed. Requests, delays,
User-Agent, and status names are unchanged.

Reads:  <data-dir>/giant_component_nodes.txt
Writes: <data-dir>/subreddit_descriptions.tsv (INDEX, SUBREDDIT, STATUS, DESCRIPTION),
        <data-dir>/fetch_descriptions.log
"""

import argparse
import logging
import os
import time
from collections import defaultdict

import requests

logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (compatible; SubredditDescriptionFetcher/1.0)"

INPUT_FILE = "giant_component_nodes.txt"
OUTPUT_FILE = "subreddit_descriptions.tsv"
LOG_FILE = "fetch_descriptions.log"

DELAY_BETWEEN_REQUESTS = 0.1  # 100ms between each request


def setup_logging(log_path):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        force=True,
    )


def fetch_subreddit_description(subreddit_name):
    """
    Fetch description for a single subreddit using public JSON API.

    Returns:
        tuple: (status, description)
        status: 'success_with_desc', 'success_no_desc', 'not_found', 'forbidden',
                'rate_limited', 'error_XXX'
        description: the description text or None
    """
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
            logger.warning(f"HTTP {response.status_code} for {subreddit_name}")
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

    except requests.exceptions.Timeout:
        logger.warning(f"Timeout for {subreddit_name}")
        return 'error_timeout', None
    except requests.exceptions.RequestException as e:
        logger.warning(f"Request error for {subreddit_name}: {e}")
        return 'error_request', None
    except Exception as e:
        logger.warning(f"Unexpected error for {subreddit_name}: {e}")
        return 'error_unknown', None


def load_subreddit_names(path):
    logger.info(f"Loading subreddit names from {path}")
    with open(path, 'r') as f:
        subreddits = [line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(subreddits)} subreddit names")
    return subreddits


def fetch_all_descriptions(subreddit_names, output_file, resume_from=None, delay=DELAY_BETWEEN_REQUESTS):
    """
    Fetch descriptions for all subreddits, one request every `delay` seconds.

    Args:
        subreddit_names: list of subreddit names
        resume_from: index to resume from (for interrupted runs); appends to the output file
    """
    stats = defaultdict(int)

    start_idx = resume_from if resume_from else 0
    total = len(subreddit_names)

    logger.info(f"Starting fetch from index {start_idx} / {total}")

    mode = 'a' if resume_from else 'w'
    with open(output_file, mode, encoding='utf-8') as f:
        if not resume_from:
            f.write("INDEX\tSUBREDDIT\tSTATUS\tDESCRIPTION\n")

        for idx, subreddit_name in enumerate(subreddit_names[start_idx:], start=start_idx):
            status, description = fetch_subreddit_description(subreddit_name)
            stats[status] += 1

            desc_clean = (description or '').replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
            f.write(f"{idx+1}\t{subreddit_name}\t{status}\t{desc_clean}\n")
            f.flush()

            time.sleep(delay)

            if (idx + 1) % 100 == 0:
                logger.info(f"Progress: {idx + 1}/{total} ({100*(idx+1)/total:.1f}%)")
                logger.info(f"Stats so far: {dict(stats)}")

    logger.info(f"Completed! Total processed: {total}")
    logger.info(f"Final statistics: {dict(stats)}")

    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data-dir', default='../data/redditwalk')
    parser.add_argument('--resume-from', type=int, default=None,
                        help='0-based index to resume from; appends to the existing TSV')
    args = parser.parse_args(argv)

    setup_logging(os.path.join(args.data_dir, LOG_FILE))
    logger.info("=" * 60)
    logger.info("Reddit Subreddit Description Fetcher")
    logger.info("=" * 60)

    subreddit_names = load_subreddit_names(os.path.join(args.data_dir, INPUT_FILE))
    output_file = os.path.join(args.data_dir, OUTPUT_FILE)
    stats = fetch_all_descriptions(subreddit_names, output_file, resume_from=args.resume_from)

    logger.info(f"Output file: {output_file}")
    return stats


if __name__ == "__main__":
    main()

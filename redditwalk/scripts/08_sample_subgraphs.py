#!/usr/bin/env python3
"""Step 08 (report 6.3, Table 16): sample RedditWalk subgraphs with random walk with restart.

Needs <data-dir>/giant_component.csv from the missing steps 05-06 (see README).

Report calls:
    --num-subgraphs 500 --size 200 --base-seed <unknown>
    --num-subgraphs 300 --size 400 --base-seed <unknown>
    --num-subgraphs 200 --size 800 --base-seed 8

Writes: <data-dir>/giant_preprocess.csv (next to the CSV),
        <out-dir>/{N}_reddit_subgraphs_{size}.pt, ..._jaccard.json, ..._nodes.json
"""

import argparse
import os

from redditwalk.dataset import RedditDataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data-dir', default='../data/redditwalk')
    parser.add_argument('--csv', default=None, help='default: <data-dir>/giant_component.csv')
    parser.add_argument('--out-dir', default=None, help='default: <data-dir>/subgraphs')
    parser.add_argument('--num-subgraphs', type=int, required=True)
    parser.add_argument('--size', type=int, required=True)
    parser.add_argument('--base-seed', type=int, required=True)
    parser.add_argument('--p-restart', type=float, default=0.3)
    args = parser.parse_args(argv)

    csv_path = args.csv or os.path.join(args.data_dir, 'giant_component.csv')
    out_dir = args.out_dir or os.path.join(args.data_dir, 'subgraphs')

    RedditDataset(stage='train', root=out_dir, creation_mode=True, csv_path=csv_path,
                  num_subgraphs=args.num_subgraphs, subgraph_size=args.size,
                  base_seed=args.base_seed, p_restart=args.p_restart)

    dataset_file = f'{args.num_subgraphs}_reddit_subgraphs_{args.size}.pt'
    D = RedditDataset(root=out_dir, dataset_file=dataset_file)
    return D.compute_M_max()


if __name__ == '__main__':
    main()

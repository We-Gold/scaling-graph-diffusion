"""
Train + evaluate Scale-MGD on one dataset (report sec. 5.3: Tables 14-15, Figs 8-9 first versions).

    uv run python scripts/run.py --config configs/ce_query_edges.yaml --dataset planar
    uv run python scripts/run.py --config configs/smoke.yaml --dataset toy --set train.epochs=2

`--dataset all` runs every dataset of the config in sequence (stops on the first error).
"""

import argparse

from scale_mgd.config import load_config
from scale_mgd.pipeline import run_experiment


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="YAML config, e.g. configs/ce_query_edges.yaml")
    p.add_argument("--dataset", required=True, help="zinc250k | planar | ego | toy | all")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="Override a config value, e.g. --set train.epochs=1")
    p.add_argument("--run-dir", default=None, help="Output dir (default outputs/<config>/<dataset>_<time>)")
    args = p.parse_args(argv)

    cfg = load_config(args.config, args.set)
    names = list(cfg.datasets) if args.dataset == "all" else [args.dataset]
    if args.dataset == "all" and args.run_dir:
        p.error("--run-dir cannot be used with --dataset all")
    for name in names:
        result, run_dir = run_experiment(cfg, name, run_dir=args.run_dir)
        print(f"Done: {name} -> {run_dir}")


if __name__ == "__main__":
    main()

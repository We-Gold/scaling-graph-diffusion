"""
Regenerate report Figs 8-9 (figsize 5x3, 64 samples) from a trained checkpoint.

From a run made by scripts/run.py (reads its config.yaml, writes <run-dir>/viz/regen/):
    uv run python scripts/regen_plots.py --run-dir outputs/ce_query_edges/planar_2026-10-06_10-00-00

From an old Turing checkpoint of the MQP repo (no saved config):
    uv run python scripts/regen_plots.py --config configs/ce_query_edges.yaml --dataset ego \
        --checkpoint /path/to/gnn_query_edges__ce__ego/checkpoints/model_final.pt --out-dir figs/ego
"""

import argparse

from scale_mgd.config import load_config
from scale_mgd.pipeline import regen_from_run_dir, regen_plots


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", help="Run dir written by scripts/run.py")
    p.add_argument("--config", help="Config YAML (with --checkpoint)")
    p.add_argument("--dataset", help="Dataset name (with --checkpoint)")
    p.add_argument("--checkpoint", help="model_final.pt (with --config and --dataset)")
    p.add_argument("--out-dir", help="Where to write the plots (with --checkpoint)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = p.parse_args(argv)

    if args.run_dir:
        regen_from_run_dir(args.run_dir, args.set)
    elif args.checkpoint and args.config and args.dataset and args.out_dir:
        cfg = load_config(args.config, args.set)
        regen_plots(cfg, args.dataset, args.checkpoint, args.out_dir)
    else:
        p.error("give --run-dir, or --config, --dataset, --checkpoint and --out-dir")


if __name__ == "__main__":
    main()

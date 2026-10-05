"""
Table 15 "Reference" rows: MMD between the train and test splits (the best a perfect model can do).

    uv run python scripts/reference_mmd.py --dataset ego
    uv run python scripts/reference_mmd.py --dataset planar --out outputs/reference_mmd_planar.json

Uses the default loaders only (Planar split seed 0, Ego split seed 1234). The report's Planar
Reference row mixes values from this loader and from the SPECTRE loader (see README).
"""

import argparse

from scale_mgd.config import package_root
from scale_mgd.pipeline import reference_mmd


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="planar | ego | zinc250k | toy")
    p.add_argument("--out", default=None, help="Output JSON (default outputs/reference_mmd_<dataset>.json)")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)
    out = args.out or package_root() / "outputs" / f"reference_mmd_{args.dataset}.json"
    reference_mmd(args.dataset, out, device_name=args.device)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

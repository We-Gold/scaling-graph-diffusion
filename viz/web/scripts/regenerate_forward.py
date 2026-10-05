"""Regenerate the forward (noise) steps of the viz sample. NEW DATA, not MQP output.

Why: the original forward files (made by SparseDiff_repo/src/gradual_process.py on
origin/sparse-diff-new-algo) called the one-step get_Qt(beta) with the cumulative
alpha_bar and chained it every step, so step 1 was already the stationary prior.

What this script writes, for t = 1..T, in the same format as the original files:
    nodes (n, 8)    float32  q(x_t | x_0) for each node, all-zero rows for padding
    edges (n, n, 5) float32  q(e_t | e_0) for each node pair, all-zero on the diagonal
                             and for padding pairs
with the DiGress closed form (marginal transition):
    Qbar_t = alpha_bar_t I + (1 - alpha_bar_t) 1 m^T,   so   q(x_t | x_0) = alpha_bar_t x_0 + (1 - alpha_bar_t) m
alpha_bar_t is the DiGress discrete cosine schedule (PredefinedNoiseScheduleDiscrete, s = 0.008),
T = 500. m are the DiGress MOSES marginals: MOSESinfos.node_types / edge_types, normalized
like DiGressDiscrete does (x_marginals = node_types / node_types.sum()). These match the
stationary distribution stored in the original step_500_noisy.npz.

Input: step_00_original.npz (index arrays, unchanged) for x_0, and the node mask of the
original step_01_noisy.npz (padding rows are all zero). The script is deterministic (no
sampling here). The backend draws one graph per step with a fixed seed.

Run from viz/web:  uv run --project backend python scripts/regenerate_forward.py
"""
import argparse
from pathlib import Path

import numpy as np

T = 500
# DiGress MOSESinfos (src/datasets/moses_dataset.py), atom order C, N, S, O, F, Cl, Br, H
MOSES_NODE_TYPES = np.array([0.722338, 0.13661, 0.163655, 0.103549, 0.1421803, 0.005411, 0.00150, 0.0])
# Edge classes: none, single, double, triple, aromatic
MOSES_EDGE_TYPES = np.array([0.89740, 0.0472947, 0.062670, 0.0003524, 0.0486])


def cosine_alpha_bar(timesteps: int, s: float = 0.008) -> np.ndarray:
    """alphas_bar of DiGress PredefinedNoiseScheduleDiscrete('cosine', timesteps), index t = 0..T."""
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    alphas = 1 - np.clip(betas, 0, 0.9999)
    return np.exp(np.cumsum(np.log(alphas)))


def step_filename(t: int) -> str:
    return f"step_{t:02d}_noisy.npz"  # same names as the original files (02d, 3 digits from 100)


def main() -> None:
    default_dir = Path(__file__).resolve().parents[1] / "data" / "digress_moses_sample" / "noise_process" / "raw"
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-dir", type=Path, default=default_dir, help="noise_process/raw folder (read and written)")
    args = ap.parse_args()

    orig = np.load(args.raw_dir / "step_00_original.npz")
    x0, e0 = orig["nodes"].astype(int), orig["edges"].astype(int)
    node_mask = np.load(args.raw_dir / "step_01_noisy.npz")["nodes"].sum(-1) > 0.5
    n = len(x0)

    m_x = MOSES_NODE_TYPES / MOSES_NODE_TYPES.sum()
    m_e = MOSES_EDGE_TYPES / MOSES_EDGE_TYPES.sum()
    x0_onehot = np.eye(len(m_x))[x0] * node_mask[:, None]
    pair_mask = node_mask[:, None] & node_mask[None, :] & ~np.eye(n, dtype=bool)
    e0_onehot = np.eye(len(m_e))[e0] * pair_mask[..., None]

    alpha_bar = cosine_alpha_bar(T)
    for t in range(1, T + 1):
        a = alpha_bar[t]
        nodes = a * x0_onehot + (1 - a) * m_x * node_mask[:, None]
        edges = a * e0_onehot + (1 - a) * m_e * pair_mask[..., None]
        np.savez(args.raw_dir / step_filename(t), nodes=nodes.astype(np.float32), edges=edges.astype(np.float32))

    print(f"wrote {T} files to {args.raw_dir}")
    print(f"alpha_bar: t=1 {alpha_bar[1]:.4f}, t=100 {alpha_bar[100]:.4f}, t=250 {alpha_bar[250]:.4f}, t=500 {alpha_bar[500]:.2e}")


if __name__ == "__main__":
    main()

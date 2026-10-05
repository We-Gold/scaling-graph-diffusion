# Diffusion animations

Five matplotlib scripts that simulate toy discrete diffusion on random tokens or a small
random graph (no trained model) and save an mp4.

| Script | Shows |
| --- | --- |
| `digress.py` | DiGress-style uniform noise on 40 tokens (3D token vs time) |
| `digress_graph.py` | DiGress-style noise on node and edge tokens of a 10-node graph, plus a graph panel |
| `hybrid.py` | Masking plus DiGress-style flips on 40 tokens |
| `md4.py` | MD4 masking with a cosine schedule, plus a theoretical vs empirical unmasking panel |
| `md4_graph.py` | MD4-style masking on a 10-node graph (3 edge types), plus a graph panel |

## Report anchors

None (kept by team decision). Topic: Sec 2.6.1 (DiGress) and 2.6.2 (masked diffusion).

## Setup

uv workspace member `viz-animations` (matplotlib, numpy, networkx). Needs `ffmpeg` on PATH
(`brew install ffmpeg` or `apt install ffmpeg`; on Turing try `module load ffmpeg`, not checked).

## Run (local)

```
cd viz/animations
uv run python md4_graph.py                         # outputs/md4_graph.mp4, 100 frames, dpi 300, about 1 min
uv run python md4_graph.py --frames 10 --dpi 72    # quick preview
for s in digress digress_graph hybrid md4 md4_graph; do uv run python $s.py; done
```

Options: `--frames` (also the number of diffusion steps), `--dpi`, `--seed` (default 0),
`--out` (default `outputs/<script>.mp4`), `--show` (open a window too).

## Run (Turing)

Not needed. If you want to, run the same commands in an interactive job with `MPLBACKEND=Agg`.

## Data

None. Each script draws its own toy data from `--seed`.

## Results

mp4 files in `outputs/` (gitignored).

## Missing

Nothing known.

## Notes

- Source: `diff-viz/` on `origin/diffusion-geometric-visualization` (a8e3e1c).
  `md4.py` and `md4_graph.py` are the newer `md4 copy.py` and `md4-graph copy.py`; the older
  versions were dropped.
- The original scripts did not seed the toy data, so old renders cannot be reproduced exactly.

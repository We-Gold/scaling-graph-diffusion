# DiGress timestep viewer

Web app that shows a molecule graph at each of the 500 DiGress steps on MOSES.
Tab 1 has a slider and play button for the reverse process (denoising, one generated
molecule) and the forward process (noising, one MOSES training molecule). Tab 2 explains
the data format. The backend draws each step as an SVG with RDKit.

## Report anchors

None (kept by team decision). Topic: Sec 2.6.1, DiGress with the marginal transition
`Q_t = alpha_bar_t I + (1 - alpha_bar_t) 1 m^T`.

## Setup

- Python: the backend is the uv workspace member `viz-web-backend` (FastAPI, RDKit, numpy).
- Node >= 20 and npm (tested with Node 24). On Turing: `module load node-js/22.14.0`.

## Run (local)

```
# Terminal 1: backend
cd viz/web/backend
uv run uvicorn viz_backend.main:app --port 8000

# Terminal 2: frontend
cd viz/web/frontend
npm ci
npm run dev        # open http://localhost:3000
```

The first backend start takes about 15 s (RDKit import).

| Env var | Where | Default | Use |
| --- | --- | --- | --- |
| `NEXT_PUBLIC_API_URL` | frontend | page host, port 8000 | Backend on another host or port. Inlined at build time, so set it before `npm run dev` or `npm run build` |
| `VIZ_DATA_DIR` | backend | `viz/web/data/digress_moses_sample` | Another sample folder with the same layout |
| `VIZ_CORS_ORIGINS` | backend | (none) | Extra allowed origins, comma separated. `localhost:3000` and `127.0.0.1:3000` are always allowed |

API: `/timestep/info`, `/timestep/{noise|denoise}/{t}/svg`, `/timestep/{noise|denoise}/{t}/data`, t = 0..500.

## Run (Turing)

No SLURM job is needed. Start both servers on the login node (or in an interactive job),
then forward the ports from your laptop and open `http://localhost:3000`:

```
ssh -L 3000:localhost:3000 -L 8000:localhost:8000 <user>@turing.wpi.edu
```

If `npm run dev` fails with `ENOSPC` (file watcher limit), use `npm run build && npm run start`.
`npm run build` downloads the Geist font from Google, so it needs network (login node is fine).

## Data

`data/digress_moses_sample/` is committed (1002 small npz files, about 0.3 MB in git).
This is an exception to the "no data in git" rule.

| Folder | Files | Content | Origin |
| --- | --- | --- | --- |
| `denoise_process/raw/` | `step_00_fully_noisy.npz`, `step_{01..500}_denoised.npz` | Index arrays: `nodes` (26,), `edges` (26, 26), padding = -1 | Original MQP output |
| `noise_process/raw/step_00_original.npz` | 1 | Index arrays of the clean molecule (MOSES train index 0) | Original MQP output |
| `noise_process/raw/step_{01..500}_noisy.npz` | 500 | `nodes` (26, 8), `edges` (26, 26, 5) float32: `q(x_t \| x_0)` | **Regenerated**, not MQP output (see below) |

Atom classes (DiGress MOSES `atom_decoder`): C, N, S, O, F, Cl, Br, H. Bond classes: none,
single, double, triple, aromatic. 20 of the 26 node slots are real atoms.

The original files were made by `SparseDiff_repo/src/gradual_process.py` on
`origin/sparse-diff-new-algo` (Hien Pham), from a DiGress model trained on MOSES
(the checkpoint is not recorded). The forward and reverse molecules are different graphs.

**Regenerated forward steps.** The original forward steps were wrong: `gradual_process.py`
passed the cumulative `alpha_bar_t` to the one-step `get_Qt` and chained it, so the graph was
already at the stationary prior at t = 1. `scripts/regenerate_forward.py` rewrites
`step_{01..500}_noisy.npz` from `step_00_original.npz` with the DiGress closed form
`q(x_t | x_0) = alpha_bar_t x_0 + (1 - alpha_bar_t) m`: discrete cosine schedule (s = 0.008),
T = 500, and the MOSES marginals `m` from DiGress `MOSESinfos.node_types` / `edge_types`,
normalized (the same `m` as the stationary distribution of the original files, so not uniform).
It is deterministic (byte-identical on rerun). To rerun:

```
cd viz/web
uv run --project backend python scripts/regenerate_forward.py
```

The backend draws one graph per forward step with a fixed seed and the same uniforms for
every t (inverse CDF). So each frame is a sample of `q(x_t | x_0)`, and the picture changes
gradually as t grows.

## Results

None. The app is a viewer only.

## Missing

- The DiGress MOSES checkpoint that produced the reverse samples.

## Notes

- Sources: `diffusion-visualization/` and `digress_moses_sample.zip` on `origin/hienpham`
  (80ddf1b). PNG renders from the zip, Docker files, and start scripts were dropped.
- Next was bumped from 15.5.4 to 15.5.27 (React 19.1.9) for security fixes.
  `npm audit` still lists Next's bundled postcss (build time only).
- Atom labels were wrong in the original code (S showed as O, O as F). Fixed.

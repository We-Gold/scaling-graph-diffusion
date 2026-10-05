"""Merge profiler CSVs (and old MQP text logs) into the layout of report tables 3-8.

    uv run python -m report --raw out/raw --out out/tables
    uv run python -m report --log Scale-MGD=../../results/tables03-08_scale_mgd_profile.txt --out out/logs

Writes `tables03-08_summary.csv` (one row per table, stage, model) and `tableNN.md` per graph type,
with rows in report order (Scale-MGD, SparserDiff, DiGress, SparseDiff, MG-Diff) and columns
Time (s), GPU (MB), RAM (MB) as mean±std over seeds (numpy std, ddof 0, NaN skipped).
Python 3.9 syntax, numpy only.
"""

import argparse
import csv
import glob
import os
import re

import numpy as np

from configs import CONFIGS, CONFIGS_BY_NAME, CONFIGS_BY_OLD_NAME

MODEL_ORDER = ["Scale-MGD", "SparserDiff", "DiGress", "SparseDiff", "MG-Diff"]
# profiler / log stage name -> report stage name
STAGE_NAMES = {
    "Noise Addition": "Forward Diffusion",
    "Model Pass": "Model Pass",
    "Training Step": "Training Step",
    "Backward Process": "Sampling",
}
METRICS = [("time", "Time (s)", 4), ("gpu", "GPU (MB)", 2), ("ram", "RAM (MB)", 2)]
SUMMARY_COLUMNS = [
    "table",
    "config",
    "stage",
    "model",
    "steps",
    "n_seeds",
    "time_mean",
    "time_std",
    "gpu_mean",
    "gpu_std",
    "ram_mean",
    "ram_std",
    "source",
]


def _float(x):
    return float(x) if x not in ("", None) else np.nan


def summarize_csv(path):
    """Mean and std over seeds for each (config, stage, model) in one profiler CSV."""
    groups = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            key = (r["config"], r["stage"], r["model"], int(r["steps"]))
            g = groups.setdefault(key, {"time": [], "gpu": [], "ram": []})
            g["time"].append(_float(r["time_s"]))
            g["gpu"].append(_float(r["gpu_mb"]))
            g["ram"].append(_float(r["ram_mb"]))

    out = []
    for (config, stage, model, steps), g in groups.items():
        row = {
            "table": CONFIGS_BY_NAME[config].table,
            "config": config,
            "stage": STAGE_NAMES[stage],
            "model": model,
            "steps": steps,
            "source": os.path.basename(path),
        }
        valid = [i for i, t in enumerate(g["time"]) if not np.isnan(t)]
        row["n_seeds"] = len(valid)
        for key in ("time", "gpu", "ram"):
            vals = [g[key][i] for i in valid]
            row[key + "_mean"] = float(np.mean(vals)) if vals else np.nan
            row[key + "_std"] = float(np.std(vals)) if vals else np.nan
        out.append(row)
    return out


_PROFILING = re.compile(r"^--- Profiling (.+) ---$")
_STEPS = re.compile(r"^Profiling with (\d+) steps over (\d+) seeds")
_AGG = re.compile(
    r"^(Noise Addition|Model Pass|Training Step|Backward Process)\s*\|"
    r"\s*([\d.naN]+)\s*(?:±|\+/-)\s*([\d.naN]+)\s*\|"
    r"\s*([\d.naN]+)\s*(?:±|\+/-)\s*([\d.naN]+)\s*\|"
    r"\s*([\d.naN]+)\s*(?:±|\+/-)\s*([\d.naN]+)\s*$"
)


def parse_log(path, model):
    """Read the 'Aggregated Results' lines of an old-format text log (column order Time, RAM, GPU)."""
    out = []
    config = steps = n_seeds = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            m = _PROFILING.match(line)
            if m:
                name = m.group(1)
                cfg = CONFIGS_BY_OLD_NAME.get(name) or CONFIGS_BY_NAME[name]
                config = cfg.name
                continue
            m = _STEPS.match(line)
            if m:
                steps, n_seeds = int(m.group(1)), int(m.group(2))
                continue
            m = _AGG.match(line.strip())
            if m and config is not None:
                v = [float(x) for x in m.groups()[1:]]
                out.append(
                    {
                        "table": CONFIGS_BY_NAME[config].table,
                        "config": config,
                        "stage": STAGE_NAMES[m.group(1)],
                        "model": model,
                        "steps": steps,
                        "n_seeds": n_seeds,
                        "time_mean": v[0],
                        "time_std": v[1],
                        "ram_mean": v[2],
                        "ram_std": v[3],
                        "gpu_mean": v[4],
                        "gpu_std": v[5],
                        "source": os.path.basename(path),
                    }
                )
    return out


def _sort_key(row):
    stages = list(STAGE_NAMES.values())
    model_rank = MODEL_ORDER.index(row["model"]) if row["model"] in MODEL_ORDER else len(MODEL_ORDER)
    return (row["table"], stages.index(row["stage"]), model_rank, row["model"])


def fmt(mean, std, digits):
    if np.isnan(mean):
        return "n/a"
    return f"{mean:.{digits}f}±{std:.{digits}f}"


def table_markdown(rows, table):
    cfg = [c for c in CONFIGS if c.table == table][0]
    lines = [
        f"Table {table}: {cfg.name} (N = {cfg.n_nodes}, dX = {cfg.dx}, dE = {cfg.de})",
        "",
        "| Stage | Model | " + " | ".join(label for _, label, _ in METRICS) + " |",
        "| --- | --- | " + " | ".join("---" for _ in METRICS) + " |",
    ]
    for r in rows:
        cells = [fmt(r[k + "_mean"], r[k + "_std"], d) for k, _, d in METRICS]
        lines.append(f"| {r['stage']} | {r['model']} | " + " | ".join(cells) + " |")
    steps = []
    for r in rows:  # rows are already in model order
        if (r["model"], r["steps"]) not in steps:
            steps.append((r["model"], r["steps"]))
    seeds = ", ".join(str(n) for n in sorted({r["n_seeds"] for r in rows}))
    note = "Diffusion steps: " + ", ".join(f"{m} {s}" for m, s in steps) + f". Seeds per row: {seeds}."
    if any(np.isnan(r["time_mean"]) for r in rows):
        note += " n/a: every seed failed (CUDA out of memory in the original logs)."
    lines += ["", note, ""]
    return "\n".join(lines)


def write_report(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = sorted(rows, key=_sort_key)
    path = os.path.join(out_dir, "tables03-08_summary.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")
    for table in sorted({r["table"] for r in rows}):
        md = table_markdown([r for r in rows if r["table"] == table], table)
        md_path = os.path.join(out_dir, f"table{table:02d}.md")
        with open(md_path, "w") as f:
            f.write(md)
        print(f"\n{md}")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default=None, help="folder with profiler CSVs (all *.csv are read)")
    p.add_argument("--csv", nargs="*", default=[], help="profiler CSV files")
    p.add_argument("--log", nargs="*", default=[], help="old text logs as MODEL=path")
    p.add_argument("--out", default="out/tables", help="output folder")
    args = p.parse_args()

    files = list(args.csv)
    if args.raw:
        files += sorted(glob.glob(os.path.join(args.raw, "*.csv")))
    rows = []
    for path in files:
        rows += summarize_csv(path)
    for item in args.log:
        model, path = item.split("=", 1)
        rows += parse_log(path, model)
    if not rows:
        p.error("no input: pass --raw, --csv or --log")
    write_report(rows, args.out)


if __name__ == "__main__":
    main()

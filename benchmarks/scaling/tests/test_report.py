"""report.py: the committed logs reproduce report cells, and CSV merging works."""

import csv
import os

import numpy as np

import report
from common import CSV_COLUMNS

RESULTS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results")


def _cell(rows, table, stage, model):
    [r] = [r for r in rows if r["table"] == table and r["stage"] == stage and r["model"] == model]
    return r


def test_scale_mgd_log_matches_report():
    rows = report.parse_log(os.path.join(RESULTS, "tables03-08_scale_mgd_profile.txt"), "Scale-MGD")
    assert len(rows) == 24
    r = _cell(rows, 3, "Training Step", "Scale-MGD")
    assert (r["time_mean"], r["time_std"]) == (2.4667, 0.0832)
    assert (r["gpu_mean"], r["gpu_std"], r["ram_mean"], r["ram_std"]) == (1729.68, 30.43, 1125.68, 8.44)
    r = _cell(rows, 6, "Sampling", "Scale-MGD")
    assert (r["time_mean"], r["gpu_mean"], r["ram_mean"], r["steps"]) == (20.4319, 2215.11, 1127.43, 20)
    # Table 3 forward diffusion: the report (0.0004±0.0001) drops the seed-10 warmup outlier
    r = _cell(rows, 3, "Forward Diffusion", "Scale-MGD")
    assert (r["time_mean"], r["time_std"]) == (0.0076, 0.0216)


def test_digress_and_mg_diff_logs_match_table6():
    rows = report.parse_log(os.path.join(RESULTS, "tables03-08_digress_profile.txt"), "DiGress")
    r = _cell(rows, 6, "Model Pass", "DiGress")
    assert (r["time_mean"], r["time_std"], r["gpu_mean"], r["ram_mean"]) == (1.5184, 0.0260, 30692.67, 1331.48)
    assert _cell(rows, 6, "Sampling", "DiGress")["gpu_mean"] == 31048.12
    assert np.isnan(_cell(rows, 6, "Training Step", "DiGress")["time_mean"])  # OOM, as in the report
    rows = report.parse_log(os.path.join(RESULTS, "tables03-08_mg_diff_profile.txt"), "MG-Diff")
    r = _cell(rows, 6, "Sampling", "MG-Diff")
    assert (r["time_mean"], r["gpu_mean"], r["steps"]) == (19.2283, 38025.09, 10)
    assert _cell(rows, 3, "Training Step", "MG-Diff")["gpu_mean"] == 2179.29  # GPU column matches the report


def test_csv_merge(tmp_path):
    path = tmp_path / "x.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for seed, t in [(10, 1.0), (23, 3.0), (27, float("nan"))]:
            w.writerow({"model": "DiGress", "config": "Large Graph", "steps": 10, "seed": seed,
                        "stage": "Backward Process", "time_s": t, "gpu_mb": 5.0, "ram_mb": 7.0})
    [r] = report.summarize_csv(str(path))
    assert (r["table"], r["stage"], r["n_seeds"]) == (6, "Sampling", 2)
    assert (r["time_mean"], r["time_std"]) == (2.0, 1.0)
    out = report.write_report([r], str(tmp_path / "out"))
    assert out and (tmp_path / "out" / "table06.md").exists()

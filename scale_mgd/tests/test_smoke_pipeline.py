"""Tiny CPU train + sample + eval + regen with configs/smoke.yaml (no network)."""

import json
import math

import pytest

from scale_mgd.config import load_config
from scale_mgd.data import Zinc250kDataset, edge_type_marginals, training_collate_fn
from scale_mgd.pipeline import regen_from_run_dir, run_experiment

VLB_OVERRIDES = ["diffusion.loss=vlb", "eval.sampler=posterior", "plots.edge_count_sampler=posterior"]
MMD_KEYS = [f"{m}_mmd{s}" for m in ("degree", "cluster", "spectre", "rbf") for s in ("", "_test")]


@pytest.mark.parametrize("overrides", [[], VLB_OVERRIDES], ids=["ce", "vlb"])
def test_train_eval_regen_toy(tmp_path, smoke_config_path, toy_datasets, overrides):
    cfg = load_config(smoke_config_path, overrides)
    result, run_dir = run_experiment(cfg, "toy", run_dir=tmp_path / "run", datasets=toy_datasets)

    saved = json.loads((run_dir / "result.json").read_text())
    for k in MMD_KEYS:
        assert k in saved
    assert saved["loss_mode"] == cfg.diffusion.loss
    assert math.isfinite(saved["final_loss"])
    assert (run_dir / "checkpoints" / "model_final.pt").exists()
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "viz" / "edge_count_dist.png").exists()
    assert (run_dir / "eval" / "triangle_distribution.png").exists()
    assert (run_dir / "eval" / "sparsediff_metrics_test.json").exists()

    out = regen_from_run_dir(run_dir)
    assert (out / "edge_count_dist.png").exists()
    assert (out / "edge_count_dist.pdf").exists()
    assert (out / "triangle_distribution.png").exists()
    assert (out / "triangle_metrics.txt").exists()


def test_zinc_tiny_molecule_metrics(tmp_path, smoke_config_path, tiny_zinc_root):
    train = Zinc250kDataset("train", root=str(tiny_zinc_root))
    test = Zinc250kDataset("test", root=str(tiny_zinc_root))
    assert len(train) == 24 and len(test) == 3
    # Second load uses the processed cache
    assert len(Zinc250kDataset("train", root=str(tiny_zinc_root))) == 24

    cfg = load_config(smoke_config_path)
    result, run_dir = run_experiment(cfg, "zinc250k", run_dir=tmp_path / "zrun",
                                     datasets={"train": train, "test": test})
    assert 0.0 <= result["mol_validity"] <= 1.0
    assert (run_dir / "eval" / "mol_metrics.json").exists()


def test_collate_and_marginals(toy_datasets):
    ds = toy_datasets["train"]
    N = ds.max_length
    X, E, E_mask, n_real = training_collate_fn([ds[0], ds[1]], N, 12, node_pad=1, edge_pad=(1, N, N))
    assert X.shape == (2, N) and E.shape == (2, 12, 3)
    real = E[E_mask]
    assert (real[:, 0] < real[:, 1]).all() and (real[:, 2] == 2).all()
    pad = E[~E_mask]
    assert (pad[:, 0] == N).all() and (pad[:, 2] == 1).all()
    probs = edge_type_marginals(ds, 12, len(ds))
    assert probs[0] == 0 and abs(probs.sum().item() - 1) < 1e-6


@pytest.mark.network
def test_planar_download_smoke(tmp_path, smoke_config_path, monkeypatch):
    monkeypatch.setenv("SGD_DATA_ROOT", str(tmp_path / "data"))
    cfg = load_config(smoke_config_path)
    result, _ = run_experiment(cfg, "planar", run_dir=tmp_path / "prun")
    assert "degree_mmd_test" in result

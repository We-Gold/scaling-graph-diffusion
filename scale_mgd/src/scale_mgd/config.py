"""
Run configuration: dataclasses, YAML loading, `--set key=value` overrides, and path helpers.

Two named configs live in `scale_mgd/configs/`:
- `ce_query_edges.yaml` (default): the 2026-03-03 runs behind report Figs 8-9 and Tables 14-15.
- `vlb_query_edges.yaml`: the VLB setting of the report text (sec. 4.1.2-4.1.6, Alg. 2-3).
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


@dataclass
class ModelConfig:
    name: str = "query_edges"  # key of scale_mgd.models.MODEL_REGISTRY
    hidden_size: int = 128
    gnn_layers: int = 4  # L_mp
    gnn_iterations: int = 5  # L_enc
    decoder_layers: int = 4
    dropout: float = 0.2
    edge_fraction: float | None = 0.3  # query_edges only


@dataclass
class DiffusionConfig:
    T: int = 100
    loss: str = "ce"  # "ce" or "vlb"
    # CE mode: Bernoulli masking with alphas = linspace(ce_alpha_start, ce_alpha_end, T).
    # The heuristic samplers use these alphas in both modes.
    ce_alpha_start: float = 0.9999
    ce_alpha_end: float = 0.0001
    # VLB mode: absorbing + uniform schedule (report Eq. 29)
    att_1: float = 0.99999
    att_T: float = 0.000001
    ctt_1: float = 0.000001
    ctt_T: float = 0.99999
    atomic_edges: bool = True
    aux_loss_weight: float = 1e-4
    adaptive_aux: bool = True


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    grad_clip: float = 1.0
    log_interval: int = 50
    use_edge_type_marginals: bool = True


@dataclass
class EvalConfig:
    num_samples: int = 64
    batch_size: int = 32
    temperature: float = 0.5
    sampler: str = "heuristic"  # "heuristic" (predict x0, re-mask) or "posterior" (Alg. 3)
    # Edge slots used at generation for Table 15 / Fig 9. The original code hardcoded 100.
    # null means "use the dataset M_max".
    gen_edge_slots: int | None = 100
    skip_rbf: bool = False
    # True reproduces the original Table 14 computation, where every ZINC bond became SINGLE.
    legacy_single_bonds: bool = False
    ref_splits: list[str] = field(default_factory=lambda: ["train", "test"])
    max_ref_graphs: int = 1000


@dataclass
class PlotsConfig:
    # Fig 8 sampler: "heuristic_argmax" (original, used for the report figure) or "posterior"
    edge_count_sampler: str = "heuristic_argmax"
    edge_count_num_samples: int = 20  # in-run plot
    regen_num_samples: int = 64  # scripts/regen_plots.py (report figures)
    figsize: list[float] = field(default_factory=lambda: [5.0, 3.0])


@dataclass
class DatasetConfig:
    M_max: int
    batch_size: int = 64
    train_fraction: float | None = None  # first fraction of the train split; null = all


@dataclass
class Config:
    name: str = "ce_query_edges"
    seed: int = 0
    device: str = "auto"
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    plots: PlotsConfig = field(default_factory=PlotsConfig)
    datasets: dict[str, DatasetConfig] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SECTIONS = {
    "model": ModelConfig,
    "diffusion": DiffusionConfig,
    "train": TrainConfig,
    "eval": EvalConfig,
    "plots": PlotsConfig,
}


def _build(cls, data: dict[str, Any]):
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**data)


def config_from_dict(raw: dict[str, Any]) -> Config:
    raw = dict(raw)
    kwargs: dict[str, Any] = {}
    for key, cls in _SECTIONS.items():
        if key in raw:
            kwargs[key] = _build(cls, raw.pop(key) or {})
    if "datasets" in raw:
        kwargs["datasets"] = {
            name: _build(DatasetConfig, d) for name, d in (raw.pop("datasets") or {}).items()
        }
    for key in ("name", "seed", "device"):
        if key in raw:
            kwargs[key] = raw.pop(key)
    if raw:
        raise ValueError(f"Unknown top-level config keys: {sorted(raw)}")
    cfg = Config(**kwargs)
    if cfg.diffusion.loss not in ("ce", "vlb"):
        raise ValueError(f"diffusion.loss must be 'ce' or 'vlb', got {cfg.diffusion.loss!r}")
    if cfg.eval.sampler not in ("heuristic", "posterior"):
        raise ValueError(f"eval.sampler must be 'heuristic' or 'posterior', got {cfg.eval.sampler!r}")
    if cfg.plots.edge_count_sampler not in ("heuristic_argmax", "posterior"):
        raise ValueError("plots.edge_count_sampler must be 'heuristic_argmax' or 'posterior'")
    return cfg


def apply_overrides(raw: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply `a.b.c=value` overrides (value parsed as YAML) to a nested dict."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        key, value = item.split("=", 1)
        node = raw
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return raw


def load_config(path: str | os.PathLike, overrides: list[str] | None = None) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    raw.setdefault("name", Path(path).stem)
    return config_from_dict(apply_overrides(raw, overrides))


def save_config(cfg: Config, path: str | os.PathLike) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)


# ---------------------------------------------------------------------------
# Paths, device, seed
# ---------------------------------------------------------------------------

def repo_root() -> Path:
    """Repo root: the first parent of this file whose pyproject.toml defines the uv workspace."""
    for parent in Path(__file__).resolve().parents:
        pp = parent / "pyproject.toml"
        if pp.exists() and "[tool.uv.workspace]" in pp.read_text():
            return parent
    return Path.cwd()


def data_root() -> Path:
    """`$SGD_DATA_ROOT` if set, else `<repo root>/data`."""
    env = os.environ.get("SGD_DATA_ROOT")
    return Path(env) if env else repo_root() / "data"


def package_root() -> Path:
    """The `scale_mgd/` part folder (holds configs/, scripts/, outputs/)."""
    return Path(__file__).resolve().parents[2]


def select_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """The original runs set no seed. This is new, so results will not match bit for bit."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


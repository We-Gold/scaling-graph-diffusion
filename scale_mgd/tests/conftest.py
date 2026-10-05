from pathlib import Path

import pytest
import torch

PART_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture
def smoke_config_path():
    return PART_ROOT / "configs" / "smoke.yaml"


@pytest.fixture
def toy_datasets():
    from scale_mgd.data import ToyGraphDataset

    return {"train": ToyGraphDataset("train"), "test": ToyGraphDataset("test")}


TINY_SMILES = [
    "CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCCC", "C1CCCCC1", "CC(C)O", "OCCO", "CC=O", "CNC",
    "CCOC", "c1ccncc1", "CC#N", "CCCl", "CCBr", "CC(=O)N", "COC", "CCS", "C=CC", "CCCO",
    "CC(C)C", "c1ccoc1", "NCCN", "CCF", "OC(=O)C", "CCC=O", "CN(C)C", "C1CC1", "CCOCC", "c1ccsc1",
]


@pytest.fixture
def tiny_zinc_root(tmp_path):
    """30-row raw.csv in the Kaggle format (quoted SMILES ending with a newline)."""
    root = tmp_path / "zinc250k"
    root.mkdir()
    lines = ["smiles,logP,qed,SAS"] + [f'"{s}\n",0.0,0.5,2.0' for s in TINY_SMILES]
    (root / "raw.csv").write_text("\n".join(lines) + "\n")
    return root

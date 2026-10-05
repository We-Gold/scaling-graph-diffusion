"""Bond type mapping in reconstruct_molecule (Table 14 metrics)."""

import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT

from scale_mgd.metrics import reconstruct_molecule


class _Vocab:
    types = {"MASK": 0, "PAD": 1, "C": 2, "O": 3}
    bonds = {"MASK": 0, "NO_BOND": 1, BT.SINGLE: 2, BT.DOUBLE: 3, BT.TRIPLE: 4, BT.AROMATIC: 5}


def _formaldehyde():
    # C=O, with the edge stored as a ZINC-style BondType key.
    X = torch.tensor([2, 3])
    E = torch.tensor([[0, 1, _Vocab.bonds[BT.DOUBLE]]])
    return X, E


def test_double_bond_kept_by_default():
    mol = reconstruct_molecule(*_formaldehyde(), _Vocab(), sanitize=True)
    assert Chem.MolToSmiles(mol) == "C=O"


def test_legacy_single_bonds():
    mol = reconstruct_molecule(*_formaldehyde(), _Vocab(), sanitize=True, legacy_single_bonds=True)
    assert Chem.MolToSmiles(mol) == "CO"

"""
Molecule metrics for report Table 14 (ZINC250k validity, uniqueness, novelty).

Known issue kept from the original code: for ZINC250k the bond vocabulary keys are RDKit
BondType objects, but `reconstruct_molecule` compares them to the strings "DOUBLE", "TRIPLE",
"AROMATIC". So every generated bond is added as a SINGLE bond. See README "Known issues".
"""

import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT

_SPECIAL_TOKENS = {"MASK", "PAD", "UNK", "CLS", "SEP"}


def reconstruct_molecule(X, E, dataset, sanitize=False, ignore_mask=True):
    """
    Build an RDKit molecule from node types X (N,) and edge slots E (M, 3) = [u, v, type].
    PAD nodes (and MASK nodes if ignore_mask) are skipped. MASK / NO_EDGE slots and duplicate
    edges are skipped. Returns an RDKit Mol, or None if RDKit fails.
    """
    idx_to_atom = {v: k for k, v in dataset.types.items()}
    idx_to_bond = {v: k for k, v in dataset.bonds.items()}

    mol = Chem.RWMol()
    atom_map = {}
    for i, atom_type in enumerate(X):
        atom_type = atom_type.item() if torch.is_tensor(atom_type) else atom_type
        if atom_type == dataset.types.get("PAD", -999):
            continue
        if ignore_mask and atom_type == dataset.types.get("MASK", 0):
            continue

        atom_symbol = idx_to_atom.get(atom_type, "C")
        if atom_symbol in _SPECIAL_TOKENS:
            atom_symbol = "C"
        try:
            atom_map[i] = mol.AddAtom(Chem.Atom(atom_symbol))
        except Exception:
            atom_map[i] = mol.AddAtom(Chem.Atom("C"))

    added_bonds = set()
    mask_bond = dataset.bonds.get("MASK", 0)
    no_bond = dataset.bonds.get("NO_BOND", dataset.bonds.get("PAD", 1))

    for edge in E:
        u, v, bond_type = edge[0].item(), edge[1].item(), edge[2].item()
        if u not in atom_map or v not in atom_map:
            continue
        if bond_type == mask_bond or bond_type == no_bond:
            continue
        bond_key = tuple(sorted([atom_map[u], atom_map[v]]))
        if bond_key in added_bonds:
            continue

        bond_label = idx_to_bond.get(bond_type, "SINGLE")
        if bond_label in ["PAD", "MASK", "NO_BOND"]:
            continue

        # Original behavior: string comparison (see module docstring).
        rd_bond = BT.SINGLE
        if bond_label == "DOUBLE":
            rd_bond = BT.DOUBLE
        elif bond_label == "TRIPLE":
            rd_bond = BT.TRIPLE
        elif bond_label == "AROMATIC":
            rd_bond = BT.AROMATIC

        try:
            mol.AddBond(atom_map[u], atom_map[v], rd_bond)
            added_bonds.add(bond_key)
        except Exception:
            pass

    try:
        mol = mol.GetMol()
        if sanitize:
            Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def mol_to_smiles(mol):
    """Canonical SMILES, or None if the molecule fails sanitization."""
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def canonical_smiles_set(smiles_list):
    """Canonicalize training SMILES (the raw Kaggle strings are not canonical and end with a newline)."""
    RDLogger.DisableLog("rdApp.*")
    out = set()
    for s in smiles_list:
        m = Chem.MolFromSmiles(str(s).strip())
        if m is not None:
            out.add(Chem.MolToSmiles(m, canonical=True))
    RDLogger.EnableLog("rdApp.*")
    return out


def molecule_metrics(raw_tensors, dataset, train_smiles=None):
    """
    Validity, uniqueness, novelty of generated molecules.

    raw_tensors: list of (x_np, e_np) from the sampler. train_smiles: canonical SMILES set for
    novelty (None means novelty is NaN). Fix vs the original: the original compared canonical
    generated SMILES to raw training strings, which inflates novelty.
    """
    total = len(raw_tensors)
    valid_smiles = []
    for x_np, e_np in raw_tensors:
        mol = reconstruct_molecule(
            torch.from_numpy(x_np).long(), torch.from_numpy(e_np).long(), dataset,
            sanitize=True, ignore_mask=True,
        )
        smi = mol_to_smiles(mol)
        if smi is not None:
            valid_smiles.append(smi)

    valid_count = len(valid_smiles)
    unique = set(valid_smiles)
    validity = valid_count / total if total > 0 else 0.0
    uniqueness = len(unique) / valid_count if valid_count > 0 else float("nan")
    if train_smiles is not None and unique:
        novelty = sum(1 for s in unique if s not in train_smiles) / len(unique)
    else:
        novelty = float("nan")

    return {
        "validity": validity,
        "uniqueness": uniqueness,
        "novelty": novelty,
        "valid_count": valid_count,
        "total": total,
    }

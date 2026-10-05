"""Build an RDKit molecule from atom and bond dicts.

Extracted from MoleculeVisualizer.create_molecule_from_graph_data in the old
diffusion_data_processor.py (origin/hienpham). The rest of that file was dropped.
"""
from typing import Dict, List, Optional

from rdkit import Chem

BOND_TYPES = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    4: Chem.BondType.AROMATIC,
}


def graph_to_mol(atoms_data: List[Dict], bonds_data: List[Dict]) -> Optional[Chem.Mol]:
    """Return an unsanitized Mol, or None on error. Noisy graphs are often invalid."""
    try:
        mol = Chem.EditableMol(Chem.Mol())
        for atom in atoms_data:
            mol.AddAtom(Chem.Atom(atom["symbol"]))
        for bond in bonds_data:
            if bond["bond_type"] in BOND_TYPES:
                mol.AddBond(bond["begin_atom"], bond["end_atom"], BOND_TYPES[bond["bond_type"]])
        return mol.GetMol()
    except Exception:
        return None

"""ZINC250k loader (in-house, from the MQP repo src/datasets/zinc250k_dataset.py)."""

import os

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from torch.utils.data import Dataset


class Zinc250kDataset(Dataset):
    """Simple dataset wrapper for Zinc250K CSV files.

    Expects a CSV at <root>/raw.csv with columns: smiles, logP, qed, SAS

    Report sec. 5.3 (Table 14, Figs 8a/9a). Split: 80/10/10 with RandomState(42).

    Attributes:
    - types: dict(atom_symbol -> index)  (0 reserved for MASK, 1 for PAD)
    - charges: dict(formal_charge_int -> index) (0 reserved for MASK, 1 for PAD)
    - bonds: dict(BondType or special string -> index) (0 reserved for MASK, 1 for NO_BOND)
    - max_length: maximum number of atoms (used by the sampler/model)
    - data: pandas DataFrame for the chosen split (column CAN_SMILES)

    The dataset returns raw per-molecule (A, C, E) in __getitem__, and exposes
    a `collate_fn` to build batched (A, C, E) tensors padded to max_length.
    """

    def __init__(self, stage="train", root=None):
        if root is None:
            from ..config import data_root

            root = str(data_root() / "zinc250k")
        assert stage in ["train", "val", "test"]
        self.stage = stage
        self.root = root

        # If a processed file already exists for this split, load it immediately
        # and skip any CSV/raw processing. This gives priority to .pt caches when
        # present (user requested behavior).
        processed_path = os.path.join(root, f"processed_{stage}.pt")
        if os.path.exists(processed_path):
            print(f"Loading cached processed dataset from {processed_path}")
            try:
                loaded = torch.load(processed_path, weights_only=False)
                if isinstance(loaded, dict) and all(
                    k in loaded for k in ("A", "C", "E")
                ):
                    self.A_tensor = loaded["A"]
                    self.C_tensor = loaded["C"]
                    self.E_tensor = loaded["E"]
                    # restore vocabs if present
                    if "types" in loaded:
                        self.types = loaded["types"]
                    if "charges" in loaded:
                        self.charges = loaded["charges"]
                    if "bonds" in loaded:
                        self.bonds = loaded["bonds"]
                    if "max_length" in loaded:
                        self.max_length = int(loaded["max_length"])
                    # restore data if available, otherwise use an empty DataFrame
                    if "data" in loaded and isinstance(loaded["data"], pd.DataFrame):
                        self.data = loaded["data"]
                    else:
                        self.data = pd.DataFrame(columns=["CAN_SMILES"])

                    # reconstruct _records from tensors for compatibility with __getitem__
                    self._records = []
                    for i in range(self.A_tensor.size(0)):
                        arow = self.A_tensor[i].tolist()
                        crow = self.C_tensor[i].tolist()
                        try:
                            n = next(
                                (
                                    j
                                    for j, v in enumerate(arow)
                                    if v == self.types.get("PAD", 1)
                                ),
                                len(arow),
                            )
                        except Exception:
                            n = len(arow)
                        A_list = arow[:n]
                        C_list = crow[:n]
                        E_mat = self.E_tensor[i][:n, :n].cpu().numpy()
                        self._records.append((A_list, C_list, E_mat))
                    return
            except Exception as e:
                print(f"Failed to load processed dataset: {e}, will reprocess.")
                # if loading fails, fall back to CSV/raw processing below
                pass

        csv_path = os.path.join(root, "raw.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Expected {csv_path} to exist. ZINC250k has no download code: get the Kaggle file "
                "250k_rndm_zinc_drugs_clean_3.csv and save it as raw.csv (see data/README.md)."
            )

        print(f"Processing raw dataset from {csv_path}")

        df = pd.read_csv(csv_path)
        # rename smiles to CAN_SMILES for compatibility with evaluation script
        if "smiles" in df.columns:
            df = df.rename(columns={"smiles": "CAN_SMILES"})

        # If split files already exist (train/val/test), prefer loading them so
        # repeated runs use the same split. Otherwise perform a deterministic
        # 80/10/10 split from the raw CSV and save the split files.
        train_path = os.path.join(root, "train.csv")
        val_path = os.path.join(root, "val.csv")
        test_path = os.path.join(root, "test.csv")

        if (
            os.path.exists(train_path)
            and os.path.exists(val_path)
            and os.path.exists(test_path)
        ):
            # load pre-split files
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            test_df = pd.read_csv(test_path)
            # ensure CAN_SMILES column exists
            for subdf in (train_df, val_df, test_df):
                if "smiles" in subdf.columns and "CAN_SMILES" not in subdf.columns:
                    subdf.rename(columns={"smiles": "CAN_SMILES"}, inplace=True)
        else:
            # deterministic split: 80/10/10
            n = len(df)
            rng = np.random.RandomState(42)
            perm = rng.permutation(n)
            n_train = int(0.8 * n)
            n_val = int(0.1 * n)

            train_idx = perm[:n_train]
            val_idx = perm[n_train : n_train + n_val]
            test_idx = perm[n_train + n_val :]

            train_df = df.iloc[train_idx].reset_index(drop=True)
            val_df = df.iloc[val_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)

            # write split files for reproducibility
            os.makedirs(root, exist_ok=True)
            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)
            test_df.to_csv(test_path, index=False)

        if stage == "train":
            self.data = train_df.reset_index(drop=True)
        elif stage == "val":
            self.data = val_df.reset_index(drop=True)
        else:
            self.data = test_df.reset_index(drop=True)

        # Build vocabularies from the entire raw CSV to keep consistent mappings
        mols = [Chem.MolFromSmiles(s) for s in df["CAN_SMILES"].astype(str).to_numpy()]
        atom_set = set()
        charge_set = set()
        for m in mols:
            if m is None:
                continue
            for a in m.GetAtoms():
                atom_set.add(a.GetSymbol())
                charge_set.add(a.GetFormalCharge())

        atom_list = sorted(atom_set)
        charge_list = sorted(charge_set)

        # types mapping: reserve 0 for MASK, 1 for PAD
        self.types = {"MASK": 0, "PAD": 1}
        idx = 2
        for a in atom_list:
            if a in self.types:
                continue
            self.types[a] = idx
            idx += 1

        # charges mapping: reserve two special integer keys for mask/pad (-999, -998)
        self.charges = {-999: 0, -998: 1}
        idx = 2
        for c in charge_list:
            self.charges[int(c)] = idx
            idx += 1

        # bonds mapping: use rdkit BondType entries as keys for real bonds
        # reserve 0 for MASK, 1 for NO_BOND
        self.bonds = {"MASK": 0, "NO_BOND": 1}
        idx = 2
        for bt in [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]:
            self.bonds[bt] = idx
            idx += 1

        # Precompute per-molecule representation for this split
        self._records = []
        max_n = 0
        for s in self.data["CAN_SMILES"].astype(str).values:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                # keep an empty placeholder (will be filtered out by training/eval)
                self._records.append(([], [], np.zeros((0, 0), dtype=np.int64)))
                continue

            n_atoms = mol.GetNumAtoms()
            max_n = max(max_n, n_atoms)

            A = [
                self.types.get(a.GetSymbol(), self.types["PAD"]) for a in mol.GetAtoms()
            ]
            C = [
                self.charges.get(a.GetFormalCharge(), self.charges[0])
                for a in mol.GetAtoms()
            ]

            E = np.full(
                (n_atoms, n_atoms), fill_value=self.bonds["NO_BOND"], dtype=np.int64
            )
            for b in mol.GetBonds():
                i = b.GetBeginAtomIdx()
                j = b.GetEndAtomIdx()
                bt = b.GetBondType()
                bond_idx = self.bonds.get(bt, self.bonds["NO_BOND"])
                E[i, j] = bond_idx
                E[j, i] = bond_idx

            self._records.append((A, C, E))

        # max length for padding (use maximum observed in entire CSV)
        self.max_length = int(max_n)

        # Build full tensors for this split and save to processed_{stage}.pt for
        # faster subsequent loading. We store A,C,E tensors and vocabularies.
        B = len(self._records)
        L = self.max_length
        A_tensor = torch.full((B, L), fill_value=self.types["PAD"], dtype=torch.long)
        C_tensor = torch.full((B, L), fill_value=self.charges[-998], dtype=torch.long)
        E_tensor = torch.full(
            (B, L, L), fill_value=self.bonds["NO_BOND"], dtype=torch.long
        )

        for i, (A, C, E) in enumerate(self._records):
            if len(A) == 0:
                continue
            n = len(A)
            A_tensor[i, :n] = torch.tensor(A, dtype=torch.long)
            C_tensor[i, :n] = torch.tensor(C, dtype=torch.long)
            E_tensor[i, :n, :n] = torch.tensor(E, dtype=torch.long)

        try:
            torch.save(
                {
                    "A": A_tensor,
                    "C": C_tensor,
                    "E": E_tensor,
                    "types": self.types,
                    "charges": self.charges,
                    "bonds": self.bonds,
                    "max_length": int(self.max_length),
                    "data": self.data,
                },
                processed_path,
            )
        except Exception:
            # If saving fails, ignore: dataset will still work from memory
            pass

        # keep tensors on the instance for potential fast access
        self.A_tensor = A_tensor
        self.C_tensor = C_tensor
        self.E_tensor = E_tensor

    def __len__(self):
        return len(self._records)

    def __getitem__(self, idx):
        A, C, E = self._records[idx]
        # Return raw (unpadded) lists/arrays; collate_fn will pad
        return A, C, E

    def collate_fn(self, batch):
        # batch: list of (A_list, C_list, E_array)
        B = len(batch)
        L = self.max_length

        A_tensor = torch.full((B, L), fill_value=self.types["PAD"], dtype=torch.long)
        C_tensor = torch.full((B, L), fill_value=self.charges[-998], dtype=torch.long)
        E_tensor = torch.full(
            (B, L, L), fill_value=self.bonds["NO_BOND"], dtype=torch.long
        )

        for i, (A, C, E) in enumerate(batch):
            if len(A) == 0:
                continue
            n = len(A)
            A_tensor[i, :n] = torch.tensor(A, dtype=torch.long)
            C_tensor[i, :n] = torch.tensor(C, dtype=torch.long)
            E_tensor[i, :n, :n] = torch.tensor(E, dtype=torch.long)

        return A_tensor, C_tensor, E_tensor


# Old name, kept so code written against the MQP repo still imports.
Zin250KDataset = Zinc250kDataset

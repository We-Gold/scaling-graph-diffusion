"""Planar loader (report Table 15, Figs 8b/9b). Auto-downloads planar_64_200.pt from SPECTRE."""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import download_url

class PlanarDataset(Dataset):
    """SPECTRE planar graphs (200 graphs, 64 nodes). Split seed 0: test 40, train 128, val 32."""

    def __init__(self, stage="train", root=None):
        if root is None:
            from ..config import data_root

            root = str(data_root() / "planar")
        self.stage = stage
        self.root = root

        # Ensure root exists
        os.makedirs(root, exist_ok=True)

        # Define types/vocab
        self.types = {"MASK": 0, "PAD": 1, "NODE": 2}  # Single node type
        self.charges = {-999: 0, -998: 1, 0: 2}        # Dummy charges
        self.bonds = {"MASK": 0, "NO_BOND": 1, "EDGE": 2} # Single edge type

        # Download if needed
        raw_file = "planar_64_200.pt"
        raw_path = os.path.join(root, raw_file)
        if not os.path.exists(raw_path):
            print(f"Downloading {raw_file}...")
            url = "https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/planar_64_200.pt"
            download_url(url, root)

        # Load Data
        # The file contains: (adjs, eigvals, eigvecs, n_nodes, max_eigval, min_eigval, same_sample, n_max)
        print(f"Loading {raw_path}...")
        try:
            # explicit weights_only=False because it loads complex objects/tuples
            data_tuple = torch.load(raw_path, weights_only=False)
            adjs = data_tuple[0]
            n_max_in_file = data_tuple[7] if len(data_tuple) > 7 else 64
        except Exception as e:
            print(f"Error loading planar dataset: {e}")
            raise e

        self.max_length = int(n_max_in_file)

        # Split logic (same as SpectreDataset)
        num_graphs = len(adjs)
        g_cpu = torch.Generator()
        g_cpu.manual_seed(0) # Same seed as spectre_dataset.py for consistency

        test_len = int(round(num_graphs * 0.2))
        train_len = int(round((num_graphs - test_len) * 0.8))
        val_len = num_graphs - train_len - test_len

        indices = torch.randperm(num_graphs, generator=g_cpu)

        if stage == "train":
            selected_indices = indices[:train_len]
        elif stage == "val":
            selected_indices = indices[train_len:train_len + val_len]
        else: # test
            selected_indices = indices[train_len + val_len:]

        self.data_adjs = [adjs[i] for i in selected_indices]

        # Precompute (A, C, E) records
        self._records = []
        for adj in self.data_adjs:
            # adj is (N, N) tensor, 0 or 1
            N_local = adj.shape[0]

            # Nodes: All generic type 2
            A = [self.types["NODE"]] * N_local

            # Charges: Dummy 0 (mapped to 2)
            C = [self.charges[0]] * N_local

            # Edges: Convert adj to dense numpy (N_local, N_local) with mapping
            # 0 -> NO_BOND (1)
            # 1 -> EDGE (2)
            E = np.full((N_local, N_local), self.bonds["NO_BOND"], dtype=np.int64)

            # Identify edges
            rows, cols = torch.where(adj > 0)
            for r, c in zip(rows, cols):
                E[r, c] = self.bonds["EDGE"]

            self._records.append((A, C, E))

    def __len__(self):
        return len(self._records)

    def __getitem__(self, idx):
        return self._records[idx]

    def collate_fn(self, batch):
        B = len(batch)
        L = self.max_length

        A_tensor = torch.full((B, L), fill_value=self.types["PAD"], dtype=torch.long)
        C_tensor = torch.full((B, L), fill_value=self.charges[-998], dtype=torch.long)
        E_tensor = torch.full((B, L, L), fill_value=self.bonds["NO_BOND"], dtype=torch.long)

        for i, (A, C, E) in enumerate(batch):
            n = len(A)
            if n > L:
                 # Should not happen if max_length is correct, but truncate just in case
                 n = L
                 A = A[:n]
                 C = C[:n]
                 E = E[:n, :n]

            A_tensor[i, :n] = torch.tensor(A, dtype=torch.long)
            C_tensor[i, :n] = torch.tensor(C, dtype=torch.long)
            E_tensor[i, :n, :n] = torch.tensor(E, dtype=torch.long)

        return A_tensor, C_tensor, E_tensor

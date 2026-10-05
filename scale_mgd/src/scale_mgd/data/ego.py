"""Ego loader (report Table 15, Figs 8c/9c). Auto-downloads Ego.pkl from the EDGE repo."""

import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import download_url
from networkx import to_numpy_array

class EgoDataset(Dataset):
    """Ego graph dataset from the EDGE repository (Facebook ego-networks).

    Downloads and processes ego graphs stored as NetworkX graphs in a pickle file.
    """
    def __init__(self, stage="train", root=None):
        if root is None:
            from ..config import data_root

            root = str(data_root() / "ego")
        self.stage = stage
        self.root = root

        # Ensure root exists
        os.makedirs(root, exist_ok=True)

        # Define types/vocab (same as planar - single node/edge type)
        self.types = {"MASK": 0, "PAD": 1, "NODE": 2}
        self.charges = {-999: 0, -998: 1, 0: 2}         # Dummy charges
        self.bonds = {"MASK": 0, "NO_BOND": 1, "EDGE": 2}

        # Download if needed
        raw_file = "Ego.pkl"
        raw_path = os.path.join(root, raw_file)
        if not os.path.exists(raw_path):
            print(f"Downloading {raw_file}...")
            url = "https://raw.githubusercontent.com/tufts-ml/graph-generation-EDGE/main/graphs/Ego.pkl"
            download_url(url, root)

        # Load Data - pickle file containing list of NetworkX graphs
        print(f"Loading {raw_path}...")
        try:
            with open(raw_path, 'rb') as f:
                networks = pickle.load(f)
            adjs = [torch.Tensor(to_numpy_array(net)).fill_diagonal_(0) for net in networks]
        except Exception as e:
            print(f"Error loading ego dataset: {e}")
            raise e

        # Compute max_length from data (no n_max stored in file)
        self.max_length = max(adj.shape[0] for adj in adjs)

        # Split logic (clean non-overlapping, seed 1234 matching spectre convention)
        num_graphs = len(adjs)
        g_cpu = torch.Generator()
        g_cpu.manual_seed(1234)

        # Match Spectre loader: train 80% of all graphs, val 20% (overlaps train in original code), test 20% disjoint.
        test_len = int(round(num_graphs * 0.2))
        train_len = int(round(num_graphs * 0.8))
        val_len = int(round(num_graphs * 0.2))

        indices = torch.randperm(num_graphs, generator=g_cpu)

        if stage == "train":
            selected_indices = indices[:train_len]
        elif stage == "val":
            selected_indices = indices[:val_len]  # note: overlaps train, mirroring Spectre
        else:  # test
            selected_indices = indices[train_len:train_len + test_len]

        self.data_adjs = [adjs[i] for i in selected_indices]

        print(f"  Split '{stage}': {len(self.data_adjs)} graphs "
              f"(total {num_graphs}, max_nodes {self.max_length})")

        # Precompute (A, C, E) records
        self._records = []
        for adj in self.data_adjs:
            N_local = adj.shape[0]

            # Nodes: All generic type 2
            A = [self.types["NODE"]] * N_local

            # Charges: Dummy 0 (mapped to 2)
            C = [self.charges[0]] * N_local

            # Edges: 0 -> NO_BOND (1), 1 -> EDGE (2)
            E = np.full((N_local, N_local), self.bonds["NO_BOND"], dtype=np.int64)
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
                n = L
                A = A[:n]
                C = C[:n]
                E = E[:n, :n]

            A_tensor[i, :n] = torch.tensor(A, dtype=torch.long)
            C_tensor[i, :n] = torch.tensor(C, dtype=torch.long)
            E_tensor[i, :n, :n] = torch.tensor(E, dtype=torch.long)

        return A_tensor, C_tensor, E_tensor

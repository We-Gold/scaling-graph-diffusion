"""Tiny in-memory graph dataset for smoke tests (no download). Same interface as PlanarDataset."""

import numpy as np
import torch
from torch.utils.data import Dataset


class ToyGraphDataset(Dataset):
    """Random connected graphs with 6 to 10 nodes and one edge type. Fixed seed per split."""

    def __init__(self, stage="train", num_graphs=8, min_nodes=6, max_nodes=10, root=None):
        self.stage = stage
        self.types = {"MASK": 0, "PAD": 1, "NODE": 2}
        self.charges = {-999: 0, -998: 1, 0: 2}
        self.bonds = {"MASK": 0, "NO_BOND": 1, "EDGE": 2}
        self.max_length = max_nodes

        rng = np.random.default_rng({"train": 0, "val": 1, "test": 2}[stage])
        self._records = []
        for _ in range(num_graphs):
            n = int(rng.integers(min_nodes, max_nodes + 1))
            E = np.full((n, n), self.bonds["NO_BOND"], dtype=np.int64)
            # A ring plus a few chords
            for i in range(n):
                j = (i + 1) % n
                E[i, j] = E[j, i] = self.bonds["EDGE"]
            for _ in range(2):
                i, j = rng.choice(n, size=2, replace=False)
                E[i, j] = E[j, i] = self.bonds["EDGE"]
            A = [self.types["NODE"]] * n
            C = [self.charges[0]] * n
            self._records.append((A, C, E))

    def __len__(self):
        return len(self._records)

    def __getitem__(self, idx):
        return self._records[idx]

    def collate_fn(self, batch):
        B, L = len(batch), self.max_length
        A_tensor = torch.full((B, L), self.types["PAD"], dtype=torch.long)
        C_tensor = torch.full((B, L), self.charges[-998], dtype=torch.long)
        E_tensor = torch.full((B, L, L), self.bonds["NO_BOND"], dtype=torch.long)
        for i, (A, C, E) in enumerate(batch):
            n = len(A)
            A_tensor[i, :n] = torch.tensor(A)
            C_tensor[i, :n] = torch.tensor(C)
            E_tensor[i, :n, :n] = torch.tensor(E)
        return A_tensor, C_tensor, E_tensor

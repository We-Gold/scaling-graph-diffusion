"""
Edge-type marginals pi_tau used by the random unmasking encoder (report sec. 4.1.4, Alg. 1).

Extracted from FullTrainingStep in the MQP repo. The counting is kept exactly as it was,
including two quirks, so that the 2026-03-03 runs are reproduced:
- every upper-triangle entry >= 1 (NO_EDGE and real edges) is counted once, then real edges
  are counted a second time, and NO_EDGE gets max(0, M_max - n_real) extra counts;
- pi_u and pi_v are never computed (the encoder samples them uniformly).
At sampling time no marginals are passed, so the encoder falls back to uniform edge types.
This train/sample mismatch is a known issue (see README).
"""

import numpy as np
import torch


def edge_type_marginals(dataset, M_max, n):
    """
    Args:
        dataset: dataset with `bonds` and `_records` (or __getitem__) giving (A, C, E_dense)
        M_max: edge slots per graph
        n: number of graphs to scan (the first n, i.e. the training subset)
    Returns:
        (num_edge_types,) probability tensor with MASK (index 0) set to 0
    """
    n_types = max(dataset.bonds.values()) + 1
    pad_idx = dataset.bonds.get("PAD", 1)
    counts = np.zeros(n_types, dtype=np.float64)

    for idx in range(n):
        item = dataset._records[idx] if hasattr(dataset, "_records") else dataset[idx]
        E_np = np.asarray(item[2])
        upper = np.triu(E_np, k=1)

        vals = upper[upper >= 1]
        vals = vals[vals < n_types].astype(np.int64)
        counts += np.bincount(vals, minlength=n_types)

        real = upper[upper > 1]
        n_real = len(real)
        real = real[real < n_types].astype(np.int64)
        counts += np.bincount(real, minlength=n_types)

        if pad_idx < n_types:
            counts[pad_idx] += max(0, M_max - n_real)

    counts[0] = 0
    counts_t = torch.tensor(counts, dtype=torch.float32)
    return counts_t / counts_t.sum()

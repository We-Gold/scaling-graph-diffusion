"""Dataset loaders (ZINC250k, Planar, Ego), sparse collate, and edge-type marginals."""

from .collate import TruncationCounter, max_true_edges_in_dataset, training_collate_fn
from .ego import EgoDataset
from .marginals import edge_type_marginals
from .planar import PlanarDataset
from .toy import ToyGraphDataset
from .zinc250k import Zin250KDataset, Zinc250kDataset

DATASETS = {
    "zinc250k": Zinc250kDataset,
    "planar": PlanarDataset,
    "ego": EgoDataset,
    "toy": ToyGraphDataset,  # smoke tests only
}

__all__ = [
    "DATASETS",
    "EgoDataset",
    "PlanarDataset",
    "ToyGraphDataset",
    "TruncationCounter",
    "Zin250KDataset",
    "Zinc250kDataset",
    "edge_type_marginals",
    "max_true_edges_in_dataset",
    "training_collate_fn",
]

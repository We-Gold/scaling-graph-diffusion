"""Scale-MGD networks (report sec. 4.1.4)."""

from .gnn import GNNEdgeDecoderLayer, SparseGNNMaskedDiffusionModel
from .query_edges import SparseGNNQueryEdgesModel

MODEL_REGISTRY = {
    "gnn": SparseGNNMaskedDiffusionModel,
    "query_edges": SparseGNNQueryEdgesModel,
}

__all__ = [
    "GNNEdgeDecoderLayer",
    "MODEL_REGISTRY",
    "SparseGNNMaskedDiffusionModel",
    "SparseGNNQueryEdgesModel",
]

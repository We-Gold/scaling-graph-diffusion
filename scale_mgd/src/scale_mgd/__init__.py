"""Scale-MGD: sparse masked graph diffusion (MQP report sec. 4.1 and 5.3)."""

from .models import SparseGNNMaskedDiffusionModel, SparseGNNQueryEdgesModel

__all__ = ["SparseGNNMaskedDiffusionModel", "SparseGNNQueryEdgesModel"]

"""Load a saved state dict into the model in context (used to regenerate Figs 8-9)."""

from pathlib import Path

import torch

from .runner import PipelineStep


def load_state_dict(model, checkpoint_path, device):
    """Strict load. Accepts a bare state dict or one wrapped in 'model_state_dict' / 'state_dict'."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        obj = obj["model_state_dict"]
    elif isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    state_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in obj.items()}
    model.load_state_dict(state_dict, strict=True)


class LoadCheckpointStep(PipelineStep):
    def __init__(self, checkpoint_path):
        super().__init__("Load checkpoint")
        self.checkpoint_path = checkpoint_path

    def execute(self, context):
        print(f"Loading checkpoint from {self.checkpoint_path}")
        load_state_dict(context["model"], self.checkpoint_path, context["device"])

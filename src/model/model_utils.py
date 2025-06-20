from collections import OrderedDict
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn


def load_model_from_checkpoint(
    checkpoint_path: str | Path, device: str = "auto"
) -> nn.Module:
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

    checkpoint_path = Path(checkpoint_path).absolute().resolve()
    config_path = checkpoint_path.parent / "config.yaml"

    config = OmegaConf.load(config_path)
    checkpoint_path = str(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model = instantiate(config.model.model).to(device)

    new_state_dict = OrderedDict()
    for key, value in checkpoint["state_dict"].items():
        new_key = key.replace("_orig_mod.", "")
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    return model

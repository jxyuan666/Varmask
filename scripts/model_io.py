"""Load DualHeadBayesianUNet from L7 checkpoint."""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Point to L7 model definition
_L7_SCRIPTS = Path(
    "/public/home/jxyuan/workplace/unet_single_dual_model/loss_log-var"
    "/models/L7_langevin_logvar/scripts"
)
if str(_L7_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_L7_SCRIPTS))

from net import DualHeadBayesianUNet  # noqa: E402


def load_l7_model(ckpt_path: str, device: str = "cpu") -> nn.Module:
    """Load DualHeadBayesianUNet from a best.pt checkpoint.

    Parameters
    ----------
    ckpt_path : str
        Path to best.pt.
    device : str
        "cuda" or "cpu".

    Returns
    -------
    nn.Module
        DualHeadBayesianUNet in eval mode with loaded weights.
    """
    model = DualHeadBayesianUNet(
        n_channels=1,
        n_classes=1,
        dropout_p=0.3,
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    return model

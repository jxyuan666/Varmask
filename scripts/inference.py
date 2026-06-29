"""Single deterministic forward pass through L7 model."""

import torch
import torch.nn as nn


def run_inference(model: nn.Module, image_tensor: torch.Tensor,
                  device: str = "cpu") -> tuple:
    """Single deterministic forward pass.  No MC Dropout.

    Parameters
    ----------
    model : nn.Module, L7 model in eval mode.
    image_tensor : torch.Tensor, (1, 1, H, W) float32.
    device : str

    Returns
    -------
    logits : torch.Tensor, (1, 1, H, W)
    log_var_raw : torch.Tensor, (1, 1, H, W)
    prob : torch.Tensor, (1, 1, H, W), sigmoid(logits)
    """
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits, log_var_raw = model(image_tensor)

    prob = torch.sigmoid(logits)

    return logits, log_var_raw, prob

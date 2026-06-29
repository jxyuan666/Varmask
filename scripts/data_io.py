"""Image loading and preprocessing."""

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


def _get_eval_transform(image_size: int = 256, image_channels: int = 1) -> A.Compose:
    """Deterministic eval transform: resize + normalize + to tensor."""
    if image_channels == 1:
        mean, std = (0.0,), (1.0,)
    else:
        mean, std = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)

    return A.Compose([
        A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(),
    ])


def load_image(image_path: str, image_size: int = 256) -> tuple:
    """Read a grayscale image and return display array + preprocessed tensor.

    Parameters
    ----------
    image_path : str
    image_size : int

    Returns
    -------
    display : np.ndarray, (H, W) uint8
    tensor : torch.Tensor, (1, 1, H, W) float32
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    display = image.copy()

    image_in = image[..., None]  # H,W → H,W,1
    transform = _get_eval_transform(image_size=image_size, image_channels=1)
    augmented = transform(image=image_in)
    tensor = augmented["image"].float().unsqueeze(0)  # (1, 1, H, W)

    if display.shape[:2] != (image_size, image_size):
        display = cv2.resize(display, (image_size, image_size),
                             interpolation=cv2.INTER_LINEAR)

    return display, tensor

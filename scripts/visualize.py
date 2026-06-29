"""Four-panel decision-stability visualization."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def plot_four_panel(
    display: np.ndarray,
    B_det: np.ndarray,
    P: np.ndarray,
    U: np.ndarray,
    latent: np.ndarray,
    output_path: str,
    title: str = "",
    dpi: int = 250,
) -> str:
    """Generate the 4-column decision-stability panel.

    Columns:
      (i)   Raw SEM image
      (ii)  Deterministic mask B_det (μ > 0)
      (iii) Boundary probability P = Φ(μ/σ)
      (iv)  Decision instability U = 4P(1-P) on B_det + latent halo

    Parameters
    ----------
    display : np.ndarray, (H, W) uint8.
    B_det : np.ndarray, (H, W) bool.
    P : np.ndarray, (H, W) float32, boundary probability.
    U : np.ndarray, (H, W) float32, decision instability.
    latent : np.ndarray, (H, W) bool.
    output_path : str.
    title : str, suptitle.
    dpi : int.

    Returns
    -------
    str, absolute path to saved PNG.
    """
    n_boundary = int(B_det.sum())
    n_latent = int(latent.sum())

    # Mask U to boundary pixels only
    U_boundary = np.full(U.shape, np.nan, dtype=np.float32)
    U_boundary[B_det] = U[B_det]

    fig, axs = plt.subplots(1, 4, figsize=(18, 5))

    # (i) Raw SEM
    axs[0].imshow(display, cmap="gray")
    axs[0].set_title("(i) Raw SEM Image", fontsize=12)
    axs[0].axis("off")

    # (ii) Deterministic mask
    axs[1].imshow(display, cmap="gray")
    axs[1].imshow(B_det, cmap="gray", alpha=0.6)
    axs[1].set_title(f"(ii) Deterministic Mask B_det\nμ>0 ({n_boundary} px)", fontsize=12)
    axs[1].axis("off")

    # (iii) Boundary probability
    im_p = axs[2].imshow(P, cmap="coolwarm", vmin=0, vmax=1)
    axs[2].set_title("(iii) Boundary Probability\nP(x) = Φ(μ/σ)", fontsize=12)
    axs[2].axis("off")
    plt.colorbar(im_p, ax=axs[2], fraction=0.046, pad=0.04)

    # (iv) Decision instability on B_det + latent halo
    axs[3].imshow(display, cmap="gray")
    im_u = axs[3].imshow(U_boundary, cmap="coolwarm", vmin=0, vmax=1)

    # Latent boundary halo (cyan)
    if n_latent > 0:
        latent_ov = np.zeros((*U.shape, 4), dtype=np.float32)
        latent_ov[latent, 1] = 0.8   # G
        latent_ov[latent, 2] = 0.8   # B
        latent_ov[latent, 3] = 0.25  # A
        axs[3].imshow(latent_ov)

    axs[3].set_title("(iv) Decision Instability on Mask\nU=4P(1-P)  cyan=latent boundary",
                     fontsize=12)
    axs[3].axis("off")
    plt.colorbar(im_u, ax=axs[3], fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return output_path

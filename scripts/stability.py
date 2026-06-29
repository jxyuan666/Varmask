"""Decision-stability computation.

Single L7 refinement → analytic boundary probability → decision instability.

Core equations:
    σ = exp(s / 2)                   logit-scale standard deviation
    P = Φ(μ / σ)                     boundary probability
    U = 4 * P * (1 - P)              decision instability
    B_det = μ > 0                    deterministic mask
"""

import numpy as np
import torch
from scipy.stats import norm

from scripts.config import VarMaskConfig
from scripts.refinement import refine_log_var


def compute_boundary_probability(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """P(x) = Φ(μ(x) / σ(x)).

    Parameters
    ----------
    mu : np.ndarray, (H, W) float32, logits.
    sigma : np.ndarray, (H, W) float32, logit-scale std.

    Returns
    -------
    np.ndarray, (H, W) float32, range [0, 1].
    """
    r = mu / (sigma + 1e-8)
    return norm.cdf(r).astype(np.float32)


def compute_decision_instability(P: np.ndarray) -> np.ndarray:
    """U(x) = 4 * P(x) * (1 - P(x)).

    Parameters
    ----------
    P : np.ndarray, (H, W) float32.

    Returns
    -------
    np.ndarray, (H, W) float32, range [0, 1].
    """
    return (4.0 * P * (1.0 - P)).astype(np.float32)


def compute_deterministic_mask(mu: np.ndarray) -> np.ndarray:
    """B_det(x) = μ(x) > 0.

    Parameters
    ----------
    mu : np.ndarray, (H, W) float32.

    Returns
    -------
    np.ndarray, (H, W) bool.
    """
    return mu > 0


def compute_latent_boundary(mu: np.ndarray, P: np.ndarray,
                            threshold: float = 0.2) -> np.ndarray:
    """Pixels not in deterministic mask but with P > threshold.

    Parameters
    ----------
    mu : np.ndarray, (H, W) float32.
    P : np.ndarray, (H, W) float32.
    threshold : float, P > this → latent boundary candidate.

    Returns
    -------
    np.ndarray, (H, W) bool.
    """
    return (mu <= 0) & (P > threshold)


def compute_decision_stability(
    logits: np.ndarray,
    log_var_raw: np.ndarray,
    cfg: VarMaskConfig,
    sample_id: str = "",
    fold_idx: int = -1,
) -> dict:
    """Full pipeline: single Langevin refinement → P, U, B_det, latent.

    Parameters
    ----------
    logits : np.ndarray, (H, W) float32.
    log_var_raw : np.ndarray, (H, W) float32.
    cfg : VarMaskConfig.
    sample_id : str.
    fold_idx : int.

    Returns
    -------
    dict with keys:
        s_refined  : np.ndarray, (H, W) float32
        sigma      : np.ndarray, (H, W) float32
        P_boundary : np.ndarray, (H, W) float32
        U_decision : np.ndarray, (H, W) float32
        B_det      : np.ndarray, (H, W) bool
        latent     : np.ndarray, (H, W) bool
    """
    # 1. Single Langevin refinement
    s_tensor = torch.from_numpy(log_var_raw).float().unsqueeze(0).unsqueeze(0)
    s_ref = refine_log_var(s_tensor, cfg, sample_id=sample_id, fold_idx=fold_idx)
    s = s_ref.squeeze().cpu().numpy()
    s = np.clip(s, cfg.logvar_min, cfg.logvar_max)

    # 2. Logit-scale std
    sigma = np.exp(0.5 * s)

    # 3. Boundary probability
    P = compute_boundary_probability(logits, sigma)

    # 4. Decision instability
    U = compute_decision_instability(P)

    # 5. Deterministic mask
    B_det = compute_deterministic_mask(logits)

    # 6. Latent boundary
    latent = compute_latent_boundary(logits, P, cfg.latent_threshold)

    return {
        "s_refined": s,
        "sigma": sigma,
        "P_boundary": P,
        "U_decision": U,
        "B_det": B_det,
        "latent": latent,
    }

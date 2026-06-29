"""L7 Langevin log-var spatial refinement.

Mirrors the refinement inside L7LangevinLogvarLoss.forward().
Uses fixed per-sample seed for reproducibility.
"""

import torch
from scripts.config import VarMaskConfig


def total_variation(x: torch.Tensor) -> torch.Tensor:
    """Pixel-wise total variation energy.

    TV(x) = mean(|x_{i+1,j} - x_{i,j}|) + mean(|x_{i,j+1} - x_{i,j}|)
    """
    dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return dh + dw


def _compute_sample_seed(cfg: VarMaskConfig, sample_id: str,
                         fold_idx: int = -1) -> int:
    """Derive a deterministic per-sample seed."""
    sid = int(sample_id) if sample_id.isdigit() else hash(sample_id) % 10000
    return cfg.random_seed + fold_idx * 10000 + sid * 100


def refine_log_var(log_var: torch.Tensor, cfg: VarMaskConfig,
                   sample_id: str = "", fold_idx: int = -1) -> torch.Tensor:
    """Langevin refinement of log_var under TV + L2 spatial prior, K steps.

    s_0 = clamp(log_var, [logvar_min, logvar_max])
    for k = 1..K:
        energy = tv_weight * TV(s_{k-1}) + l2_weight * mean(s_{k-1}^2)
        s_k = s_{k-1} - 0.5 * step_size * grad(energy)
              + sqrt(step_size) * noise_scale * eps_k
        s_k = clamp(s_k, [logvar_min, logvar_max])
    return s_K.detach()

    Parameters
    ----------
    log_var : torch.Tensor, (1, 1, H, W), raw log_var from model head.
    cfg : VarMaskConfig
    sample_id : str, used for per-sample seed derivation.
    fold_idx : int

    Returns
    -------
    torch.Tensor, (1, 1, H, W), detached, clamped, float32.
    """
    if cfg.reproducible_refinement:
        seed = _compute_sample_seed(cfg, sample_id, fold_idx)
        torch.manual_seed(seed)

    s = torch.clamp(log_var, min=cfg.logvar_min, max=cfg.logvar_max)
    s_refined = s.detach().clone()

    for _ in range(cfg.langevin_steps):
        with torch.enable_grad():
            s_refined = s_refined.detach().requires_grad_(True)

            energy = (
                cfg.tv_weight * total_variation(s_refined)
                + cfg.l2_logvar_weight * (s_refined ** 2).mean()
            )

            grad_s = torch.autograd.grad(
                energy, s_refined,
                create_graph=False, retain_graph=False,
            )[0]

            noise = torch.randn_like(s_refined)

            s_refined = (
                s_refined
                - 0.5 * cfg.langevin_step_size * grad_s
                + (cfg.langevin_step_size ** 0.5)
                * cfg.analysis_refinement_noise_scale * noise
            )

        s_refined = torch.clamp(s_refined, min=cfg.logvar_min, max=cfg.logvar_max)

    return s_refined.detach()

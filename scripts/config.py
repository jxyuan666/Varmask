"""Minimal configuration for decision-stability pipeline."""

from dataclasses import dataclass


@dataclass
class VarMaskConfig:
    """Parameters for L7 model → decision-stability computation."""

    # ── Log-var clamp range (from L7 formulation) ──
    logvar_min: float = -10.0
    logvar_max: float = 5.0

    # ── Langevin refinement (mirrors L7 training) ──
    langevin_steps: int = 5
    langevin_step_size: float = 0.05
    tv_weight: float = 0.001
    l2_logvar_weight: float = 0.0001
    analysis_refinement_noise_scale: float = 0.5
    reproducible_refinement: bool = True
    random_seed: int = 42
    per_sample_seed: bool = True

    # ── Image ──
    image_size: int = 256
    image_channels: int = 1

    # ── Latent boundary ──
    latent_threshold: float = 0.2  # P > this → candidate latent boundary

    # ── Runtime paths ──
    ckpt_path: str = ""
    raw_dir: str = ""
    output_dir: str = ""

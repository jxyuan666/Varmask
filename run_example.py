"""End-to-end example: raw SEM image → decision-stability panel.

Usage:
    cd /public/home/jxyuan/workplace/VarMask
    python run_example.py
"""
import sys
from pathlib import Path

# Ensure scripts/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib; matplotlib.use("Agg")

from scripts.config import VarMaskConfig
from scripts.model_io import load_l7_model
from scripts.data_io import load_image
from scripts.inference import run_inference
from scripts.stability import compute_decision_stability
from scripts.visualize import plot_four_panel

# ── Config ──
cfg = VarMaskConfig()
cfg.ckpt_path = (
    "/public/home/jxyuan/workplace/unet_single_dual_model/loss_log-var"
    "/models/L7_langevin_logvar/runs/fold_0/checkpoints/best.pt"
)

# ── Example image ──
example_img = str(Path(__file__).resolve().parent / "examples" / "sem_example.png")
output_path = str(Path(__file__).resolve().parent / "examples" / "output_stability.png")

if not Path(example_img).exists():
    print(f"Example image not found: {example_img}")
    print("Place a 256×256 grayscale SEM image at examples/sem_example.png")
    sys.exit(1)

# ── Pipeline ──
print("Step 1: Loading model ...")
model = load_l7_model(cfg.ckpt_path, device="cpu")

print("Step 2: Loading image ...")
display, tensor = load_image(example_img, image_size=cfg.image_size)

print("Step 3: Model forward pass ...")
logits, log_var_raw, prob = run_inference(model, tensor, "cpu")
mu = logits.squeeze().cpu().numpy()
s_raw = log_var_raw.squeeze().cpu().numpy()

print("Step 4: Computing decision stability ...")
out = compute_decision_stability(mu, s_raw, cfg, sample_id="example")

print("Step 5: Generating panel ...")
title = (
    f"VarMask Decision Stability  |  "
    f"B={out['B_det'].sum()} px  "
    f"U_mean={out['U_decision'][out['B_det']].mean():.3f}  "
    f"P_med={np.median(out['P_boundary'][out['B_det']]):.3f}  "
    f"latent={out['latent'].sum()} px"
)

plot_four_panel(
    display, out["B_det"], out["P_boundary"],
    out["U_decision"], out["latent"],
    output_path, title=title,
)

print(f"Saved: {output_path}")
print("Done.")

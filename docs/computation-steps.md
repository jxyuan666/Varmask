# VarMask — Decision-Stability Computation Steps

## Overview

VarMask converts the L7 dual-head U-Net's log-var output into a **decision-stability
map** that shows, for every pixel in the deterministic boundary mask, how stable
that boundary decision is under the model's own logit-space uncertainty.

**Key property**: no calibration, no reference set, no hand-tuned thresholds.

## Pipeline

```
Raw SEM image (grayscale, any size → resized to 256×256)
        │
        ▼
┌─────────────────────────────────────────────┐
│ Step 1 — Model forward pass (once)            │
│   DualHeadBayesianUNet.eval()                │
│   torch.no_grad()                            │
│   → μ  = logits      (H,W) float32          │
│   → s_raw = log_var  (H,W) float32          │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│ Step 2 — L7 Langevin refinement (once)       │
│   s = refine_log_var(s_raw)                 │
│   5 steps TV+L2 energy-guided dynamics      │
│   noise_scale=0.5, fixed per-image seed     │
│   → s  (H,W) float32, spatially coherent    │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│ Step 3 — Logit-scale standard deviation      │
│   σ(x) = exp(s(x) / 2)                      │
│   → σ  (H,W) float32                        │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│ Step 4 — Analytic boundary probability       │
│   P(x) = Φ(μ(x) / σ(x))                     │
│   Φ = standard normal CDF                   │
│   Derivation: z ~ N(μ, σ²) ⇒ P(z>0)=Φ(μ/σ) │
│   → P  (H,W) float32, range [0,1]           │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│ Step 5 — Decision instability                │
│   U(x) = 4 × P(x) × (1 − P(x))              │
│   U=0 at P=0 or P=1 (stable)                │
│   U=1 at P=0.5 (maximum instability)        │
│   → U  (H,W) float32, range [0,1]           │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│ Step 6 — Deterministic mask + latent         │
│   B_det(x) = μ(x) > 0                       │
│   latent(x) = μ(x) ≤ 0 and P(x) > 0.2      │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│ Step 7 — Four-panel visualization            │
│   (i)   Raw SEM image                       │
│   (ii)  Deterministic mask B_det            │
│   (iii) Boundary probability P(x)           │
│   (iv)  Decision instability U on B_det     │
│         + latent boundary halo (cyan)       │
└─────────────────────────────────────────────┘
```

## Interpretation

| Quantity | Meaning |
|----------|---------|
| P(x) = 0 | Model is certain this pixel is NOT a boundary |
| P(x) = 1 | Model is certain this pixel IS a boundary |
| P(x) ≈ 0.5 | Model cannot decide — maximum instability |
| U(x) ≈ 0 | Stable decision (P near 0 or 1) |
| U(x) ≈ 1 | Unstable decision (P near 0.5) |

**Why μ/σ?**  The model outputs μ (best guess logit) and σ (how uncertain
that guess is).  The ratio μ/σ is the signal-to-noise ratio — how many
standard deviations μ is from the decision boundary at 0.  Large |μ/σ| →
confident.  |μ/σ| near 0 → uncertain.

## Module Structure

```
VarMask/
├── scripts/
│   ├── config.py       — VarMaskConfig dataclass
│   ├── model_io.py     — load_l7_model()
│   ├── data_io.py      — load_image()
│   ├── inference.py    — run_inference()
│   ├── refinement.py   — refine_log_var() + total_variation()
│   ├── stability.py    — P, U, B_det, latent computations
│   └── visualize.py    — plot_four_panel()
├── docs/
│   └── computation-steps.md
├── examples/
└── run_example.py
```

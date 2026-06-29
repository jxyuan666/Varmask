# VarMask

Decision-stability metrology confidence from L7 dual-head U-Net.

## Pipeline

```
Raw SEM → L7 Model → μ, s_raw
         → Langevin refinement → s
         → σ = exp(s/2)
         → P = Φ(μ/σ)        boundary probability
         → U = 4P(1-P)       decision instability
         → B_det = μ > 0     deterministic mask
```

## Quick Start

```bash
python run_example.py
```

## Modules

| Module | Function |
|--------|----------|
| `scripts/config.py` | VarMaskConfig dataclass |
| `scripts/model_io.py` | load_l7_model() |
| `scripts/data_io.py` | load_image() |
| `scripts/inference.py` | run_inference() |
| `scripts/refinement.py` | refine_log_var() |
| `scripts/stability.py` | compute_decision_stability() |
| `scripts/visualize.py` | plot_four_panel() |

## Output

Four-panel figure: (i) raw SEM, (ii) deterministic mask, (iii) boundary probability P=Φ(μ/σ), (iv) decision instability U=4P(1-P) on mask with latent boundary halo.

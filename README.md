# Feature Convergence & Residual Connection Experiments

Empirical verification of the **cone effect** in neural networks and how **residual connections** mitigate feature convergence / rank collapse.

## Background

Deep neural networks exhibit a surprising inductive bias: the feature representations at intermediate layers tend to **converge to a narrow cone** in the embedding space, with pairwise cosine similarity approaching 1 as depth increases. This phenomenon (the *cone effect*) has important implications for model expressiveness and is closely related to the **rank collapse** problem in self-attention networks.

This repository provides experiments to verify these phenomena and demonstrate that **residual connections** — and more recently, **attention-based residual connections** (AttnRes) — can effectively alleviate them.

## Key References

| Paper | Key Contribution |
|-------|-----------------|
| [Mind the Gap](https://arxiv.org/abs/2203.02053) (Liang et al., NeurIPS 2022) | Demonstrates the cone effect & modality gap in multi-modal models |
| [Attention is not all you need](https://proceedings.mlr.press/v139/dong21a.html) (Dong et al., ICML 2021) | Proves pure self-attention loses rank doubly exponentially with depth; skip connections are crucial |
| [Attention Residuals](https://arxiv.org/abs/2603.15031) (Kimi Team, 2026) | Proposes replacing fixed residual accumulation with learned depth-wise attention |

## Experiments

### Exp 1: Cone Effect Verification
Reproduces **Fig.2(b)** from *Mind the Gap*. Shows that average pairwise cosine similarity of features increases rapidly with depth for MLPs with nonlinear activations.

### Exp 2: Residual Connection Comparison
Compares **Plain MLP** vs **ResidualMLP** (`h + f(h)`) vs **AttnResMLP** (attention over depth). Measures cosine similarity, effective rank, and relative residual across depth.

### Exp 3: Layer-wise Rank Collapse
Reproduces **Fig.2** from *Attention is not all you need*. Measures relative residual at each layer, showing how pure attention collapses to rank-1 while skip connections prevent this.

### Exp 4: Random Seed Sensitivity
Reproduces **Fig.2(c)** from *Mind the Gap*. Different random initializations create distinctly different cones, explaining the modality gap phenomenon.

## Usage

```bash
# Run all experiments
python run_experiments.py

# Run specific experiments
python run_experiments.py --exp 1 2

# Custom output directory
python run_experiments.py --out-dir results/figures
```

## Project Structure

```
feature-convergence/
├── run_experiments.py        # Main entry point
├── src/
│   ├── __init__.py
│   ├── models.py             # MLP, ResidualMLP, AttnResMLP
│   ├── metrics.py            # Cosine similarity, effective rank, residual norm
│   ├── experiments.py        # Core experiment logic
│   └── plotting.py           # Publication-ready visualizations
├── figures/                  # Generated figures
├── requirements.txt
└── README.md
```

## Requirements

```
numpy>=1.24
matplotlib>=3.7
scipy>=1.10
seaborn>=0.12
```

Optional (for GPU acceleration and extended experiments):
```
torch>=2.0
```

## TODO

- [ ] Add PyTorch implementation for GPU support
- [ ] Implement Block AttnRes variant
- [ ] Add Transformer-based experiments (self-attention + MLP)
- [ ] Reproduce scaling law results from AttnRes paper
- [ ] Add cosine similarity visualization for real data (MSCOCO)

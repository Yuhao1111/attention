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
| [Signal Propagation in Transformers](https://arxiv.org/abs/2206.03126) (Noci et al., NeurIPS 2022) | Analyzes rank collapse in Transformers; shows normalization and depth-scaled residuals (α=1/√L) preserve signal propagation |
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

### Exp 5: Transformer Rank Collapse vs Depth
Measures cosine similarity, effective rank, and relative residual at the output of a `TransformerEncoder` (Pre-LN, Xavier init) as depth grows from 1 to 24 blocks. Illustrates the rank-collapse dynamics analyzed theoretically in Noci et al. (2022).

### Exp 6: Normalization Mode Comparison
Compares the four normalization strategies — **none**, **Post-LN**, **Pre-LN**, and **RMSNorm** — on all three rank-collapse metrics across depth. Pre-LN and RMSNorm are expected to best preserve effective rank, consistent with findings in Noci et al. (2022).

### Exp 7: Residual Scaling α=1 vs α=1/√L
For each total depth L ∈ {2, 4, 8, 12, 16, 24, 32}, compares the standard residual (α=1) against the depth-scaled variant (α=1/√L) proposed in Noci et al. (2022) §4.
**Metric:** Pearson correlation between the vectors of pairwise cosine similarities at the input and at the output — a value near 1 means the model faithfully preserves the input similarity structure.
With α=1/√L, each layer's contribution is kept at the same relative scale regardless of total depth, preventing the residual stream from drowning out the attention signal.

## Usage

```bash
# Run all experiments
python run_experiments.py

# Run specific experiments
python run_experiments.py --exp 1 2

# Run Transformer experiments only
python run_experiments.py --exp 5 6 7

# Custom output directory
python run_experiments.py --out-dir results/figures
```

## Project Structure

```
feature-convergence/
├── run_experiments.py        # Main entry point
├── src/
│   ├── __init__.py
│   ├── models.py             # MLP, ResidualMLP, AttnResMLP, TransformerBlock/Encoder
│   ├── metrics.py            # Cosine similarity, effective rank, residual norm
│   ├── experiments.py        # Core experiment logic (exp1–exp7)
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
- [ ] Reproduce scaling law results from AttnRes paper
- [ ] Add cosine similarity visualization for real data (MSCOCO)
- [ ] Multi-head attention variant for TransformerBlock
- [ ] Reproduce Noci et al. Fig.2 (signal-to-noise ratio curves)

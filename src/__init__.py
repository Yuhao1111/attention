"""Feature Convergence & Residual Connection Experiments.

This package implements experiments to verify:
1. The "cone effect": neural network features converge to a narrow cone with depth.
2. Residual connections alleviate this convergence.
3. Attention-based residuals (AttnRes) further improve depth-wise information flow.

References:
- Liang et al., "Mind the Gap" (NeurIPS 2022): arxiv.org/abs/2203.02053
- Dong et al., "Attention is not all you need" (ICML 2021)
- Kimi Team, "Attention Residuals" (2026): arxiv.org/abs/2603.15031
"""

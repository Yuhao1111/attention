"""Core experiment functions.

Experiment 1: Cone Effect Verification
    - Initialize MLPs with different activations and depths
    - Measure how cosine similarity changes with depth
    - Reproduce Fig.2(b) from "Mind the Gap"

Experiment 2: Residual Connection Mitigation
    - Compare MLP vs ResidualMLP vs AttnResMLP
    - Measure relative residual (rank collapse indicator) across depth
    - Reproduce Fig.2 from "Attention is not all you need"

Experiment 3: Random Seed Sensitivity
    - Show that different random seeds create different cones
    - Reproduce Fig.2(c) from "Mind the Gap"

Experiment 5: Transformer Rank Collapse vs Depth
    - Measure cosine similarity, effective rank, relative residual
      for a TransformerEncoder (pre_ln) as depth grows
    - Relates to Noci et al. (2022), Theorem 1

Experiment 6: Normalization Mode Comparison
    - Compare none / post_ln / pre_ln / rmsnorm on rank collapse
    - Illustrates how layer normalization affects signal propagation

Experiment 7: Residual Scaling α=1 vs α=1/√L
    - For each total depth L, compare α=1 vs α=1/√L residual scaling
    - Measures preservation of input pairwise-similarity structure
    - See Noci et al. (2022) §4 on depth-scaled residuals
"""

import numpy as np
from typing import Dict, List, Any

from .models import MLP, ResidualMLP, AttnResMLP, TransformerEncoder, build_model, ACTIVATIONS
from .metrics import (
    cosine_similarity_stats,
    effective_rank,
    relative_residual,
    pairwise_cosine_similarity,
)


# ---------------------------------------------------------------------------
# Experiment 1: Cone Effect vs Depth & Activation
# ---------------------------------------------------------------------------

def exp1_cone_effect(
    d: int = 512,
    max_layers: int = 25,
    n_samples: int = 1000,
    activations: List[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Verify the cone effect: cosine similarity increases with depth.

    Reproduces Fig.2(b) from "Mind the Gap" (Liang et al., 2022).

    Args:
        d:           Hidden dimension (input = hidden for simplicity).
        max_layers:  Maximum number of layers to test.
        n_samples:   Number of random input vectors.
        activations: List of activation names to compare.
        seed:        Random seed for input generation.

    Returns:
        Dict mapping activation name -> list of (n_layers, avg_cosine_sim).
    """
    if activations is None:
        activations = ["relu", "sigmoid", "leaky_relu", "tanh", "linear"]

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, d))

    results = {}
    for act_name in activations:
        layer_sims = []
        for n_layers in range(1, max_layers + 1):
            model = MLP(d, d, n_layers, activation=act_name, seed=0)
            features = model.forward(X)
            stats = cosine_similarity_stats(features)
            layer_sims.append((n_layers, stats["mean"]))
        results[f"MLP+{act_name}"] = layer_sims

    return results


# ---------------------------------------------------------------------------
# Experiment 2: Residual Connection Comparison
# ---------------------------------------------------------------------------

def exp2_residual_comparison(
    d_input: int = 32,
    d_hidden: int = 32,
    max_layers: int = 25,
    n_samples: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compare rank collapse across model types.

    Measures:
    - Average cosine similarity (cone effect)
    - Effective rank
    - Relative residual (token uniformity)

    across depths for MLP, ResidualMLP, and AttnResMLP.

    Args:
        d_input:    Input dimension.
        d_hidden:   Hidden dimension.
        max_layers: Maximum depth.
        n_samples:  Number of input samples.
        seed:       Random seed.

    Returns:
        Dict mapping model_name -> dict of metric lists.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, d_input))

    model_configs = {
        "MLP (no residual)": "mlp",
        "ResidualMLP (h+f(h))": "residual_mlp",
        "AttnResMLP (attention residual)": "attnres_mlp",
    }

    results = {}
    for label, model_type in model_configs.items():
        cos_sims = []
        eff_ranks = []
        rel_residuals = []

        for n_layers in range(1, max_layers + 1):
            model = build_model(
                model_type,
                d_input=d_input,
                d_hidden=d_hidden,
                n_layers=n_layers,
                activation="relu",
                seed=0,
            )
            intermediates = model.forward(X, return_intermediates=True)

            # Use the last layer's features
            feats = intermediates[-1]

            stats = cosine_similarity_stats(feats)
            cos_sims.append(stats["mean"])
            eff_ranks.append(effective_rank(feats))
            rel_residuals.append(relative_residual(feats))

        results[label] = {
            "cos_sim": cos_sims,
            "eff_rank": eff_ranks,
            "rel_residual": rel_residuals,
        }

    return results


# ---------------------------------------------------------------------------
# Experiment 3: Layer-wise Analysis (like Dong et al. Fig.2)
# ---------------------------------------------------------------------------

def exp3_layerwise_residual(
    d_input: int = 256,
    d_hidden: int = 256,
    n_layers: int = 16,
    n_samples: int = 200,
    seed: int = 42,
) -> Dict[str, List[float]]:
    """Measure relative residual at each layer for different architectures.

    Reproduces Fig.2 from "Attention is not all you need" (Dong et al., 2021).

    Returns:
        Dict mapping model_name -> list of relative_residual per layer.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, d_input))

    results = {}
    for label, model_type in [
        ("SAN (no residual)", "mlp"),
        ("SAN + skip", "residual_mlp"),
        ("SAN + AttnRes", "attnres_mlp"),
    ]:
        model = build_model(
            model_type,
            d_input=d_input,
            d_hidden=d_hidden,
            n_layers=n_layers,
            activation="relu",
            seed=0,
        )
        intermediates = model.forward(X, return_intermediates=True)
        rr = [relative_residual(h) for h in intermediates]
        results[label] = rr

    return results


# ---------------------------------------------------------------------------
# Experiment 4: Different Random Seeds → Different Cones
# ---------------------------------------------------------------------------

def exp4_seed_sensitivity(
    d_input: int = 512,
    d_hidden: int = 512,
    n_layers: int = 5,
    n_samples: int = 500,
    n_seeds: int = 10,
) -> Dict[str, Any]:
    """Show that different random initializations produce different cones.

    Reproduces Fig.2(c) from "Mind the Gap".

    Returns:
        Dict with 'features' (list of feature arrays) and 'seeds'.
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n_samples, d_input))

    all_features = []
    seeds = list(range(n_seeds))

    for s in seeds:
        model = MLP(d_input, d_hidden, n_layers, activation="relu", seed=s)
        feats = model.forward(X)
        all_features.append(feats)

    return {"features": all_features, "seeds": seeds, "input": X}


# ---------------------------------------------------------------------------
# Experiment 5: Transformer Rank Collapse vs Depth
# ---------------------------------------------------------------------------

def exp5_transformer_rank_collapse(
    d_model: int = 256,
    max_layers: int = 24,
    n_samples: int = 300,
    norm_mode: str = "pre_ln",
    seed: int = 42,
) -> Dict[str, Any]:
    """Measure rank collapse in TransformerEncoder as depth increases.

    Tracks three complementary metrics at the final layer output:
      - Average pairwise cosine similarity  (↑ = more collapsed)
      - Effective rank                       (↓ = more collapsed)
      - Relative residual                    (↓ = more collapsed)

    Relates to Noci et al. (2022), which shows pure self-attention
    suffers doubly-exponential rank collapse without residuals/norms.

    Args:
        d_model:    Token feature dimension.
        max_layers: Maximum depth to test.
        n_samples:  Number of random input vectors.
        norm_mode:  Normalization mode passed to TransformerEncoder.
        seed:       Random seed for input generation.

    Returns:
        Dict with keys 'cos_sim', 'eff_rank', 'rel_residual', 'n_layers'.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, d_model))

    cos_sims, eff_ranks, rel_residuals = [], [], []

    for n_layers in range(1, max_layers + 1):
        model = TransformerEncoder(d_model, n_layers, norm_mode=norm_mode, seed=0)
        feats = model.forward(X)

        cos_sims.append(cosine_similarity_stats(feats)["mean"])
        eff_ranks.append(effective_rank(feats))
        rel_residuals.append(relative_residual(feats))

    return {
        "cos_sim": cos_sims,
        "eff_rank": eff_ranks,
        "rel_residual": rel_residuals,
        "n_layers": list(range(1, max_layers + 1)),
    }


# ---------------------------------------------------------------------------
# Experiment 6: Normalization Mode Comparison
# ---------------------------------------------------------------------------

def exp6_norm_comparison(
    d_model: int = 256,
    max_layers: int = 16,
    n_samples: int = 300,
    seed: int = 42,
) -> Dict[str, Dict]:
    """Compare rank-collapse behaviour across four normalization strategies.

    Tests:  none / post_ln / pre_ln / rmsnorm
    Metrics per configuration:
      - cos_sim, eff_rank, rel_residual (all vs depth)

    Args:
        d_model:    Feature dimension.
        max_layers: Maximum depth.
        n_samples:  Number of random input vectors.
        seed:       Random seed.

    Returns:
        Dict  norm_mode  ->  {'cos_sim': [...], 'eff_rank': [...],
                               'rel_residual': [...]}
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, d_model))

    results = {}
    for norm in ("none", "post_ln", "pre_ln", "rmsnorm"):
        cos_sims, eff_ranks, rel_residuals = [], [], []
        for n_layers in range(1, max_layers + 1):
            model = TransformerEncoder(d_model, n_layers, norm_mode=norm, seed=0)
            feats = model.forward(X)
            cos_sims.append(cosine_similarity_stats(feats)["mean"])
            eff_ranks.append(effective_rank(feats))
            rel_residuals.append(relative_residual(feats))
        results[norm] = {
            "cos_sim": cos_sims,
            "eff_rank": eff_ranks,
            "rel_residual": rel_residuals,
        }

    return results


# ---------------------------------------------------------------------------
# Experiment 7: Residual Scaling  α=1  vs  α=1/√L
# ---------------------------------------------------------------------------

def exp7_residual_scaling(
    d_model: int = 256,
    depths: List[int] = None,
    n_samples: int = 300,
    norm_mode: str = "none",
    seed: int = 42,
) -> Dict[str, Any]:
    """Compare α=1 vs α=1/√L residual scaling across different depths.

    For each total depth L we build two TransformerEncoders:
      - alpha_1:       α1 = α2 = 1.0  (standard residual)
      - alpha_inv_sqrt: α1 = α2 = 1/√L  (depth-scaled, Noci et al. §4)

    Metric — correlation preservation:
        corr(cos_sim(input), cos_sim(output))
    Pearson correlation between the vectors of pairwise cosine
    similarities at the input and at the output.  A value close to 1
    means the model preserves the input similarity structure.

    Also records average output cosine similarity (rank-collapse indicator).

    Args:
        d_model:    Feature dimension.
        depths:     List of depths to test (default: [2,4,8,12,16,24,32]).
        n_samples:  Number of random input vectors.
        norm_mode:  Normalization mode (default 'none' to isolate α effect).
        seed:       Random seed.

    Returns:
        Dict with:
          'depths': list of L values
          'alpha_1':        {'correlation': [...], 'cos_sim': [...]}
          'alpha_inv_sqrt': {'correlation': [...], 'cos_sim': [...]}
    """
    if depths is None:
        depths = [2, 4, 8, 12, 16, 24, 32]

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, d_model))
    input_sims = pairwise_cosine_similarity(X)

    results: Dict[str, Any] = {"depths": depths}

    for label, alpha_fn in [
        ("alpha_1",        lambda L: 1.0),
        ("alpha_inv_sqrt", lambda L: 1.0 / np.sqrt(L)),
    ]:
        corrs, cos_sims = [], []
        for L in depths:
            alpha = float(alpha_fn(L))
            model = TransformerEncoder(
                d_model, L, norm_mode=norm_mode, alpha1=alpha, alpha2=alpha, seed=0
            )
            feats = model.forward(X)

            output_sims = pairwise_cosine_similarity(feats)
            # Pearson correlation between input and output pairwise similarities
            corr = float(np.corrcoef(input_sims, output_sims)[0, 1])
            corrs.append(corr)
            cos_sims.append(cosine_similarity_stats(feats)["mean"])

        results[label] = {"correlation": corrs, "cos_sim": cos_sims}

    return results

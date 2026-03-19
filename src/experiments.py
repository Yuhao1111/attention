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
"""

import numpy as np
from typing import Dict, List, Any

from .models import MLP, ResidualMLP, AttnResMLP, build_model, ACTIVATIONS
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
    d_input: int = 512,
    d_hidden: int = 512,
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

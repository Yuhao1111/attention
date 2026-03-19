"""Visualization functions for experiment results.

All figures are saved to the `figures/` directory.
Style: clean, publication-ready, with consistent color scheme.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List, Any, Optional
from pathlib import Path

# Use non-interactive backend
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

COLORS = {
    "MLP (no residual)": "#e74c3c",
    "ResidualMLP (h+f(h))": "#2ecc71",
    "AttnResMLP (attention residual)": "#3498db",
    "SAN (no residual)": "#e74c3c",
    "SAN + skip": "#2ecc71",
    "SAN + AttnRes": "#3498db",
}

NORM_COLORS = {
    "none":     "#e74c3c",
    "post_ln":  "#e67e22",
    "pre_ln":   "#2ecc71",
    "rmsnorm":  "#3498db",
}

NORM_LABELS = {
    "none":     "No Norm",
    "post_ln":  "Post-LN",
    "pre_ln":   "Pre-LN",
    "rmsnorm":  "RMSNorm",
}

SCALE_COLORS = {
    "alpha_1":        "#e74c3c",
    "alpha_inv_sqrt": "#3498db",
}

SCALE_LABELS = {
    "alpha_1":        r"$\alpha = 1$",
    "alpha_inv_sqrt": r"$\alpha = 1/\sqrt{L}$",
}

ACTIVATION_COLORS = {
    "MLP+relu": "#e74c3c",
    "MLP+sigmoid": "#9b59b6",
    "MLP+leaky_relu": "#e67e22",
    "MLP+tanh": "#1abc9c",
    "MLP+linear": "#95a5a6",
}

LINE_STYLES = {
    "MLP+relu": "-",
    "MLP+sigmoid": "--",
    "MLP+leaky_relu": "-.",
    "MLP+tanh": ":",
    "MLP+linear": "-",
}


def _series_color(label: str) -> str:
    """Return a consistent color, including dynamic ResidualMLP labels."""
    if label.startswith("ResidualMLP"):
        return COLORS["ResidualMLP (h+f(h))"]
    return COLORS.get(label, "#333333")


def _setup_style():
    """Apply clean matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "lines.linewidth": 2.0,
        "figure.dpi": 150,
    })


def _save(fig, name: str, out_dir: str = "figures"):
    """Save figure to disk."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Fig 1: Cone Effect (cos sim vs depth, different activations)
# ---------------------------------------------------------------------------

def plot_cone_effect(
    results: Dict[str, List],
    out_dir: str = "figures",
):
    """Plot average cosine similarity vs number of layers.

    Reproduces Fig.2(b) from 'Mind the Gap'.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, data in results.items():
        layers, sims = zip(*data)
        color = ACTIVATION_COLORS.get(label, "#333333")
        ls = LINE_STYLES.get(label, "-")
        display_name = label.replace("MLP+", "n×(Linear+") + ")"
        ax.plot(layers, sims, label=display_name, color=color, linestyle=ls,
                marker="o", markersize=3)

    ax.set_xlabel("Number of layers (n)")
    ax.set_ylabel("Average cosine similarity")
    ax.set_title("Cone Effect: Feature Similarity Increases with Depth")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right")
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

    _save(fig, "exp1_cone_effect", out_dir)


# ---------------------------------------------------------------------------
# Fig 2: Residual Comparison (3 subplots)
# ---------------------------------------------------------------------------

def plot_residual_comparison(
    results: Dict[str, Dict],
    out_dir: str = "figures",
):
    """Plot cosine similarity, effective rank, and relative residual.

    Three subplots comparing MLP / ResidualMLP / AttnResMLP.
    """
    _setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metrics = [
        ("cos_sim", "Avg. Cosine Similarity", "Cosine Similarity vs Depth"),
        ("eff_rank", "Effective Rank", "Effective Rank vs Depth"),
        ("rel_residual", "Relative Residual", "Relative Residual vs Depth"),
    ]

    for ax, (metric_key, ylabel, title) in zip(axes, metrics):
        for label, data in results.items():
            values = data[metric_key]
            layers = list(range(1, len(values) + 1))
            color = _series_color(label)
            ax.plot(layers, values, label=label, color=color, marker="o",
                    markersize=3)
        ax.set_xlabel("Number of layers")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)

    fig.suptitle(
        "Residual Connections Mitigate Feature Convergence",
        fontsize=15, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save(fig, "exp2_residual_comparison", out_dir)


# ---------------------------------------------------------------------------
# Fig 3: Layer-wise Relative Residual
# ---------------------------------------------------------------------------

def plot_layerwise_residual(
    results: Dict[str, List[float]],
    out_dir: str = "figures",
):
    """Plot relative residual at each layer.

    Reproduces Fig.2 from 'Attention is not all you need' (Dong et al.).
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, rr_values in results.items():
        layers = list(range(len(rr_values)))
        color = COLORS.get(label, "#333333")
        ls = "--" if "no residual" in label else "-"
        ax.plot(layers, rr_values, label=label, color=color, linestyle=ls,
                marker="o", markersize=4)

    ax.set_xlabel("Layer index $l$")
    ax.set_ylabel(r"$\|res(X_l)\|_{1,\infty} \;/\; \|X_l\|_{1,\infty}$")
    ax.set_title("Layer-wise Relative Residual (Rank Collapse Indicator)")
    ax.legend()
    ax.set_ylim(-0.05, 1.10)

    _save(fig, "exp3_layerwise_residual", out_dir)


# ---------------------------------------------------------------------------
# Fig 4: Seed Sensitivity (PCA visualization)
# ---------------------------------------------------------------------------

def plot_seed_cones(
    data: Dict[str, Any],
    out_dir: str = "figures",
):
    """PCA visualization of features from different random seeds.

    Reproduces Fig.2(c) from 'Mind the Gap'.
    """
    _setup_style()
    features_list = data["features"]
    seeds = data["seeds"]

    # Stack all features and compute joint PCA
    all_feats = np.vstack(features_list)
    mean = all_feats.mean(axis=0)
    centered = all_feats - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ Vt[:2].T  # Project onto top-2 PCs

    # Split back
    n_per_seed = features_list[0].shape[0]
    fig, ax = plt.subplots(figsize=(8, 7))

    cmap = plt.cm.tab10
    for i, s in enumerate(seeds):
        start = i * n_per_seed
        end = start + n_per_seed
        xy = proj[start:end]
        ax.scatter(xy[:, 0], xy[:, 1], s=8, alpha=0.5,
                   color=cmap(i % 10), label=f"seed={s}")

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("Different Random Seeds → Different Cones (PCA)")
    ax.legend(markerscale=3, fontsize=8, ncol=2, loc="best")

    _save(fig, "exp4_seed_cones", out_dir)


# ---------------------------------------------------------------------------
# Fig 5: Cosine Similarity Histogram
# ---------------------------------------------------------------------------

def plot_cosine_histogram(
    features: np.ndarray,
    title: str = "Pairwise Cosine Similarity",
    label: str = "",
    out_dir: str = "figures",
    filename: str = "cosine_hist",
):
    """Histogram of pairwise cosine similarities."""
    from .metrics import pairwise_cosine_similarity

    _setup_style()
    sims = pairwise_cosine_similarity(features)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sims, bins=80, color="#3498db", alpha=0.7, edgecolor="white",
            linewidth=0.3)
    avg = np.mean(sims)
    ax.axvline(avg, color="#e74c3c", linestyle="--", linewidth=2,
               label=f"Mean={avg:.3f}")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Count")
    ax.set_title(f"{title}" + (f" ({label})" if label else ""))
    ax.legend()

    _save(fig, filename, out_dir)


# ---------------------------------------------------------------------------
# Fig 5: Transformer Rank Collapse vs Depth
# ---------------------------------------------------------------------------

def plot_transformer_rank_collapse(
    results: Dict[str, Any],
    out_dir: str = "figures",
):
    """Three-panel plot of Transformer rank collapse metrics vs depth.

    Shows cos_sim, effective rank, and relative residual as the number
    of Transformer blocks increases.  Reproduces qualitatively the
    signal-propagation analysis of Noci et al. (2022).
    """
    _setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    layers = results["n_layers"]
    color = "#9b59b6"

    panels = [
        ("cos_sim",      "Avg. Cosine Similarity",  "Cosine Similarity vs Depth"),
        ("eff_rank",     "Effective Rank",           "Effective Rank vs Depth"),
        ("rel_residual", "Relative Residual",        "Relative Residual vs Depth"),
    ]

    for ax, (key, ylabel, title) in zip(axes, panels):
        ax.plot(layers, results[key], color=color, marker="o", markersize=4)
        ax.set_xlabel("Number of Transformer blocks")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    fig.suptitle(
        "Transformer Rank Collapse vs Depth  (Noci et al., 2022)",
        fontsize=15, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save(fig, "exp5_transformer_rank_collapse", out_dir)


# ---------------------------------------------------------------------------
# Fig 6: Normalization Mode Comparison
# ---------------------------------------------------------------------------

def plot_norm_comparison(
    results: Dict[str, Dict],
    out_dir: str = "figures",
):
    """Compare none / post_ln / pre_ln / rmsnorm across three rank metrics.

    Three subplots, each with four lines (one per normalization mode).
    """
    _setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    panels = [
        ("cos_sim",      "Avg. Cosine Similarity",  "Cosine Similarity vs Depth"),
        ("eff_rank",     "Effective Rank",           "Effective Rank vs Depth"),
        ("rel_residual", "Relative Residual",        "Relative Residual vs Depth"),
    ]

    n_layers = None
    for ax, (key, ylabel, title) in zip(axes, panels):
        for norm, data in results.items():
            values = data[key]
            if n_layers is None:
                n_layers = list(range(1, len(values) + 1))
            color = NORM_COLORS.get(norm, "#333333")
            label = NORM_LABELS.get(norm, norm)
            ax.plot(n_layers, values, label=label, color=color, marker="o",
                    markersize=3)
        ax.set_xlabel("Number of Transformer blocks")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)

    fig.suptitle(
        "Effect of Normalization on Rank Collapse in Transformers",
        fontsize=15, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save(fig, "exp6_norm_comparison", out_dir)


# ---------------------------------------------------------------------------
# Fig 7: Residual Scaling  α=1  vs  α=1/√L
# ---------------------------------------------------------------------------

def plot_residual_scaling(
    results: Dict[str, Any],
    out_dir: str = "figures",
):
    """Two-panel plot comparing α=1 vs α=1/√L residual scaling.

    Left panel:  Pearson correlation between input and output pairwise
                 cosine similarities (higher = better signal preservation).
    Right panel: Average output cosine similarity (lower = less collapsed).
    """
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    depths = results["depths"]

    for label in ("alpha_1", "alpha_inv_sqrt"):
        data  = results[label]
        color = SCALE_COLORS[label]
        name  = SCALE_LABELS[label]

        axes[0].plot(depths, data["correlation"], label=name, color=color,
                     marker="o", markersize=5)
        axes[1].plot(depths, data["cos_sim"],     label=name, color=color,
                     marker="s", markersize=5)

    axes[0].set_xlabel("Depth $L$")
    axes[0].set_ylabel("Pearson Correlation (input vs output pairwise sim)")
    axes[0].set_title("Similarity Structure Preservation vs Depth")
    axes[0].set_ylim(-0.1, 1.05)
    axes[0].axhline(y=1.0, color="gray", linestyle=":", linewidth=1)
    axes[0].legend()

    axes[1].set_xlabel("Depth $L$")
    axes[1].set_ylabel("Avg. Pairwise Cosine Similarity")
    axes[1].set_title("Output Rank Collapse vs Depth")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend()

    fig.suptitle(
        r"Residual Scaling: $\alpha=1$ vs $\alpha=1/\sqrt{L}$  (Noci et al., 2022)",
        fontsize=15, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save(fig, "exp7_residual_scaling", out_dir)


# ---------------------------------------------------------------------------
# Fig 8: Init Variance Sweep
# ---------------------------------------------------------------------------

def plot_init_variance(
    results: Dict[str, Any],
    out_dir: str = "figures",
):
    """Two-panel plot: cosine similarity and effective rank vs init variance scale.

    X-axis is the variance multiplier m (where scale = m/d), log-scaled.
    Three lines per panel, one per model (MLP / ResidualMLP / AttnResMLP).

    Reference: Noci et al. (2022) Eq.17-18.
    """
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    scale_mults = results["scale_mults"]
    x_labels = [f"{m}" for m in scale_mults]

    model_labels = [
        "MLP (no residual)",
        "ResidualMLP (h+f(h))",
        "AttnResMLP (attention residual)",
    ]

    for label in model_labels:
        if label not in results:
            continue
        data = results[label]
        color = COLORS.get(label, "#333333")
        axes[0].plot(scale_mults, data["cos_sim"], label=label, color=color,
                     marker="o", markersize=5)
        axes[1].plot(scale_mults, data["eff_rank"], label=label, color=color,
                     marker="s", markersize=5)

    for ax, ylabel, title in [
        (axes[0], "Avg. Cosine Similarity",
         "Cosine Similarity vs Init Variance Scale"),
        (axes[1], "Effective Rank",
         "Effective Rank vs Init Variance Scale"),
    ]:
        ax.set_xscale("log")
        ax.set_xticks(scale_mults)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_xlabel(r"Init variance scale $m$  (scale $= m/d$,  Xavier $= 1/d$)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)

    # Mark Xavier (m=1) and He (m=2) reference lines
    for ax in axes:
        ax.axvline(x=1.0, color="gray", linestyle=":", linewidth=1,
                   label="Xavier (m=1)")
        ax.axvline(x=2.0, color="goldenrod", linestyle=":", linewidth=1,
                   label="He (m=2)")

    fig.suptitle(
        r"Effect of Init Variance on Rank Collapse  (Noci et al., 2022 Eq.17-18)",
        fontsize=15, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save(fig, "exp8_init_variance", out_dir)

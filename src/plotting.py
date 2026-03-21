"""Visualization functions for experiment results.

All figures are saved to the `figures/` directory.
Style: publication-ready — 14pt axis labels, 16pt titles, 12pt legend,
       2.5pt lines, top/right spines removed, grid alpha=0.2.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from pathlib import Path

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

COLORS = {
    # Core exp2 models
    "MLP (no residual)":                "#d62728",
    "ResidualMLP (α=0.1)":              "#2ca02c",
    "AttnRes + MLP (no self-attn)":     "#1f77b4",
    "AttnResMLP":                       "#1f77b4",
    # Extended residual variants
    "ResidualMLP (α=1)":                "#2ca02c",
    "ResidualMLP (α=1/√L)":             "#27ae60",
    "Pure Self-Attention":              "#8e44ad",
    # Exp 3 SAN labels
    "SAN (no residual)":                "#d62728",
    "SAN + skip":                       "#2ca02c",
    "SAN + AttnRes":                    "#1f77b4",
    # Exp 8 legacy labels
    "ResidualMLP (h+f(h))":             "#2ca02c",
    "AttnResMLP (attention residual)":  "#1f77b4",
}

NORM_COLORS  = {"none": "#d62728", "post_ln": "#e67e22",
                "pre_ln": "#2ca02c", "rmsnorm": "#1f77b4"}
NORM_LABELS  = {"none": "No Norm", "post_ln": "Post-LN",
                "pre_ln": "Pre-LN", "rmsnorm": "RMSNorm"}
SCALE_COLORS = {"alpha_1": "#d62728", "alpha_inv_sqrt": "#1f77b4"}
SCALE_LABELS = {"alpha_1": r"$\alpha = 1$",
                "alpha_inv_sqrt": r"$\alpha = 1/\sqrt{L}$"}

ACTIVATION_COLORS = {
    "MLP+relu":       "#d62728",
    "MLP+sigmoid":    "#9467bd",
    "MLP+leaky_relu": "#e67e22",
    "MLP+tanh":       "#17becf",
    "MLP+linear":     "#7f7f7f",
}
LINE_STYLES = {
    "MLP+relu": "-", "MLP+sigmoid": "--",
    "MLP+leaky_relu": "-.", "MLP+tanh": ":", "MLP+linear": "-",
}

TRAINING_COLORS = {
    "PlainMLP":            "#d62728",
    "ResidualMLP (α=0.1)": "#2ca02c",
    "AttnResMLP":          "#1f77b4",
}


def _series_color(label: str) -> str:
    if label.startswith("ResidualMLP"):
        return COLORS.get(label, "#2ca02c")
    return COLORS.get(label, "#333333")


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def _setup_style():
    """Publication-ready matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.grid":        True,
        "grid.alpha":       0.2,
        "grid.linestyle":   "--",
        "font.size":        12,
        "axes.labelsize":   14,
        "axes.titlesize":   16,
        "legend.fontsize":  12,
        "lines.linewidth":  2.5,
        "figure.dpi":       150,
    })


def _style_ax(ax):
    """Remove top/right spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig, name: str, out_dir: str = "figures"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Exp 1: Cone Effect
# ---------------------------------------------------------------------------

def plot_cone_effect(results: Dict[str, List], out_dir: str = "figures"):
    """Avg cosine similarity vs depth for different activations."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, data in results.items():
        layers, sims = zip(*data)
        color   = ACTIVATION_COLORS.get(label, "#333333")
        ls      = LINE_STYLES.get(label, "-")
        display = label.replace("MLP+", "n×(Linear+") + ")"
        ax.plot(layers, sims, label=display, color=color,
                linestyle=ls, marker="o", markersize=5)

    ax.set_xlabel("Number of layers (n)")
    ax.set_ylabel("Average cosine similarity")
    ax.set_title("Cone Effect: Feature Similarity Increases with Depth")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    _style_ax(ax)
    fig.tight_layout()
    _save(fig, "exp1_cone_effect", out_dir)


# ---------------------------------------------------------------------------
# Exp 2: Residual Comparison (3 models)
# ---------------------------------------------------------------------------

def plot_residual_comparison(results: Dict[str, Dict], out_dir: str = "figures"):
    """Three-panel comparison: cos_sim, eff_rank, rel_residual vs depth."""
    _setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    panels = [
        ("cos_sim",      "Avg. Cosine Similarity", "Cosine Similarity vs Depth"),
        ("eff_rank",     "Effective Rank",          "Effective Rank vs Depth"),
        ("rel_residual", "Relative Residual",       "Relative Residual vs Depth"),
    ]

    for ax, (key, ylabel, title) in zip(axes, panels):
        for label, data in results.items():
            layers = list(range(1, len(data[key]) + 1))
            ax.plot(layers, data[key], label=label,
                    color=_series_color(label), marker="o", markersize=5)
        ax.set_xlabel("Number of layers")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        _style_ax(ax)

    fig.suptitle("Residual Connections Mitigate Feature Convergence",
                 fontsize=17, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "exp2_residual_comparison", out_dir)


# ---------------------------------------------------------------------------
# Exp 2b: Alpha Sweep (ResidualMLP)
# ---------------------------------------------------------------------------

def plot_alpha_sweep(results: Dict[str, Any], out_dir: str = "figures"):
    """Effective rank vs depth for ResidualMLP across alpha values.

    Uses a plasma colormap; α=0.10 is drawn thicker to highlight the
    advisor-validated value; α=1/√L is drawn dashed (dynamic per depth).
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    alpha_labels  = results["alpha_labels"]
    n_layers_list = results["n_layers"]
    n             = len(alpha_labels)
    colors        = [plt.cm.plasma(i / (n - 1)) for i in range(n)]

    for i, label in enumerate(alpha_labels):
        if label not in results:
            continue
        ls = "--" if "1/√L" in label else "-"
        lw = 3.5 if "0.10" in label else 2.5
        ax.plot(n_layers_list, results[label]["eff_rank"],
                label=label, color=colors[i],
                linestyle=ls, linewidth=lw, marker="o", markersize=5)

    ax.set_xlabel("Number of layers")
    ax.set_ylabel("Effective Rank")
    ax.set_title("Effect of Residual Scaling α on Rank Collapse\n"
                 r"(ResidualMLP, $d=1024$, $h = h + \alpha \cdot f(h)$)")
    ax.legend(title="α value", ncol=2)
    _style_ax(ax)
    fig.tight_layout()
    _save(fig, "exp2b_alpha_sweep", out_dir)


# ---------------------------------------------------------------------------
# Exp 3: Layer-wise Relative Residual
# ---------------------------------------------------------------------------

def plot_layerwise_residual(results: Dict[str, List[float]], out_dir: str = "figures"):
    """Relative residual at each layer for different architectures."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, rr_values in results.items():
        layers = list(range(len(rr_values)))
        ls     = "--" if "no residual" in label else "-"
        ax.plot(layers, rr_values, label=label,
                color=COLORS.get(label, "#333333"),
                linestyle=ls, marker="o", markersize=5)

    ax.set_xlabel("Layer index $l$")
    ax.set_ylabel(r"$\|res(X_l)\|_{1,\infty} \;/\; \|X_l\|_{1,\infty}$")
    ax.set_title("Layer-wise Relative Residual (Rank Collapse Indicator)")
    ax.legend()
    ax.set_ylim(-0.05, 1.10)
    _style_ax(ax)
    fig.tight_layout()
    _save(fig, "exp3_layerwise_residual", out_dir)


# ---------------------------------------------------------------------------
# Exp 4a: Seed Sensitivity (PCA)
# ---------------------------------------------------------------------------

def plot_seed_cones(data: Dict[str, Any], out_dir: str = "figures"):
    """PCA visualization of features from different random seeds."""
    _setup_style()
    features_list = data["features"]
    seeds         = data["seeds"]

    all_feats = np.vstack(features_list)
    centered  = all_feats - all_feats.mean(axis=0)
    _, _, Vt  = np.linalg.svd(centered, full_matrices=False)
    proj      = centered @ Vt[:2].T

    n_per = features_list[0].shape[0]
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap    = plt.cm.tab10
    for i, s in enumerate(seeds):
        xy = proj[i * n_per: (i + 1) * n_per]
        ax.scatter(xy[:, 0], xy[:, 1], s=8, alpha=0.5,
                   color=cmap(i % 10), label=f"seed={s}")

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("Different Random Seeds → Different Cones (PCA)")
    ax.legend(markerscale=3, ncol=2)
    _style_ax(ax)
    fig.tight_layout()
    _save(fig, "exp4_seed_cones", out_dir)


# ---------------------------------------------------------------------------
# Exp 4b: Cosine Similarity Histogram
# ---------------------------------------------------------------------------

def plot_cosine_histogram(features: np.ndarray, title: str = "Pairwise Cosine Similarity",
                          label: str = "", out_dir: str = "figures",
                          filename: str = "cosine_hist"):
    from .metrics import pairwise_cosine_similarity
    _setup_style()
    sims = pairwise_cosine_similarity(features)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(sims, bins=80, color="#1f77b4", alpha=0.7, edgecolor="white", linewidth=0.3)
    avg = np.mean(sims)
    ax.axvline(avg, color="#d62728", linestyle="--", linewidth=2,
               label=f"Mean={avg:.3f}")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Count")
    ax.set_title(f"{title}" + (f" ({label})" if label else ""))
    ax.legend()
    _style_ax(ax)
    fig.tight_layout()
    _save(fig, filename, out_dir)


# ---------------------------------------------------------------------------
# Exp 5: Transformer Rank Collapse vs Depth
# ---------------------------------------------------------------------------

def plot_transformer_rank_collapse(results: Dict[str, Any], out_dir: str = "figures"):
    """Three-panel Transformer rank collapse metrics vs depth."""
    _setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    layers = results["n_layers"]
    color  = "#9467bd"

    panels = [
        ("cos_sim",      "Avg. Cosine Similarity", "Cosine Similarity vs Depth"),
        ("eff_rank",     "Effective Rank",          "Effective Rank vs Depth"),
        ("rel_residual", "Relative Residual",       "Relative Residual vs Depth"),
    ]
    for ax, (key, ylabel, title) in zip(axes, panels):
        ax.plot(layers, results[key], color=color, marker="o", markersize=5)
        ax.set_xlabel("Number of Transformer blocks")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        _style_ax(ax)

    fig.suptitle("Transformer Rank Collapse vs Depth  (Noci et al., 2022)",
                 fontsize=17, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "exp5_transformer_rank_collapse", out_dir)


# ---------------------------------------------------------------------------
# Exp 6: Normalization Mode Comparison
# ---------------------------------------------------------------------------

def plot_norm_comparison(results: Dict[str, Dict], out_dir: str = "figures"):
    """Compare none / post_ln / pre_ln / rmsnorm across three rank metrics."""
    _setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    panels = [
        ("cos_sim",      "Avg. Cosine Similarity", "Cosine Similarity vs Depth"),
        ("eff_rank",     "Effective Rank",          "Effective Rank vs Depth"),
        ("rel_residual", "Relative Residual",       "Relative Residual vs Depth"),
    ]
    n_layers = None
    for ax, (key, ylabel, title) in zip(axes, panels):
        for norm, data in results.items():
            values = data[key]
            if n_layers is None:
                n_layers = list(range(1, len(values) + 1))
            ax.plot(n_layers, values,
                    label=NORM_LABELS.get(norm, norm),
                    color=NORM_COLORS.get(norm, "#333333"),
                    marker="o", markersize=5)
        ax.set_xlabel("Number of Transformer blocks")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        _style_ax(ax)

    fig.suptitle("Effect of Normalization on Rank Collapse in Transformers",
                 fontsize=17, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "exp6_norm_comparison", out_dir)


# ---------------------------------------------------------------------------
# Exp 7: Residual Scaling α=1 vs α=1/√L
# ---------------------------------------------------------------------------

def plot_residual_scaling(results: Dict[str, Any], out_dir: str = "figures"):
    """Two-panel: similarity preservation and rank collapse vs depth."""
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
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
    _style_ax(axes[0])

    axes[1].set_xlabel("Depth $L$")
    axes[1].set_ylabel("Avg. Pairwise Cosine Similarity")
    axes[1].set_title("Output Rank Collapse vs Depth")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend()
    _style_ax(axes[1])

    fig.suptitle(
        r"Residual Scaling: $\alpha=1$ vs $\alpha=1/\sqrt{L}$  (Noci et al., 2022)",
        fontsize=17, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "exp7_residual_scaling", out_dir)


# ---------------------------------------------------------------------------
# Exp 8: Init Variance Sweep
# ---------------------------------------------------------------------------

def plot_init_variance(results: Dict[str, Any], out_dir: str = "figures"):
    """Two-panel: cosine similarity and effective rank vs init variance scale."""
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    scale_mults  = results["scale_mults"]
    x_labels     = [str(m) for m in scale_mults]
    model_labels = [
        "MLP (no residual)",
        "ResidualMLP (h+f(h))",
        "AttnResMLP (attention residual)",
    ]

    for label in model_labels:
        if label not in results:
            continue
        data  = results[label]
        color = COLORS.get(label, "#333333")
        axes[0].plot(scale_mults, data["cos_sim"], label=label,
                     color=color, marker="o", markersize=5)
        axes[1].plot(scale_mults, data["eff_rank"], label=label,
                     color=color, marker="s", markersize=5)

    for ax, ylabel, title in [
        (axes[0], "Avg. Cosine Similarity",   "Cosine Similarity vs Init Variance Scale"),
        (axes[1], "Effective Rank",           "Effective Rank vs Init Variance Scale"),
    ]:
        ax.set_xscale("log")
        ax.set_xticks(scale_mults)
        ax.set_xticklabels(x_labels, fontsize=11)
        ax.set_xlabel(r"Init variance scale $m$  (scale $= m/d$,  Xavier $= 1/d$)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.axvline(x=1.0, color="gray",      linestyle=":", linewidth=1.5)
        ax.axvline(x=2.0, color="goldenrod", linestyle=":", linewidth=1.5)
        _style_ax(ax)

    fig.suptitle(
        r"Effect of Init Variance on Rank Collapse  (Noci et al., 2022 Eq.17-18)",
        fontsize=17, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "exp8_init_variance", out_dir)


# ---------------------------------------------------------------------------
# MNIST Training Curves
# ---------------------------------------------------------------------------

def plot_training_curves(histories: Dict[str, Any], out_dir: str = "figures"):
    """Four-panel training curves: loss, accuracy, cosine similarity, eff rank."""
    _setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    panels = [
        ("train_loss", "Train Loss",            "Training Loss"),
        ("test_acc",   "Test Accuracy",          "Test Accuracy"),
        ("cos_sim",    "Avg. Cosine Similarity", "Feature Cosine Similarity (cone effect)"),
        ("eff_rank",   "Effective Rank",         "Effective Rank of Final-Layer Features"),
    ]
    for ax, (key, ylabel, title) in zip(axes, panels):
        for name, h in histories.items():
            color  = TRAINING_COLORS.get(name, "#333333")
            epochs = list(range(1, len(h[key]) + 1))
            ax.plot(epochs, h[key], label=name, color=color,
                    marker="o", markersize=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        _style_ax(ax)

    fig.suptitle("MNIST Training: PlainMLP vs ResidualMLP(α=0.1) vs AttnResMLP",
                 fontsize=17, fontweight="bold")
    fig.tight_layout()
    _save(fig, "mnist_training_curves", out_dir)


# ---------------------------------------------------------------------------
# AttnRes Attention Weight Heatmap  (Fig.8 style)
# ---------------------------------------------------------------------------

def plot_attn_heatmap(attn_before: np.ndarray, attn_after: np.ndarray,
                      n_layers: int = 8, out_dir: str = "figures"):
    """Side-by-side heatmaps before/after training (AttnRes paper Fig.8 style)."""
    _setup_style()

    def _mask(mat):
        m = mat.copy().astype(float)
        for row in range(n_layers + 1):
            m[row, row + 1:] = np.nan
        return m

    before     = _mask(attn_before)
    after      = _mask(attn_after)
    src_labels = [f"h{i}" for i in range(n_layers + 1)]
    qry_labels = [f"q{l}" for l in range(n_layers)] + ["q_out"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    vmax = max(after[~np.isnan(after)].max(), 0.01)

    for ax, mat, title in [
        (axes[0], before, "Before training  (uniform)"),
        (axes[1], after,  "After training   (learned)"),
    ]:
        im = ax.imshow(mat, aspect="auto", cmap="Blues",
                       vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_xticks(range(n_layers + 1))
        ax.set_xticklabels(src_labels, fontsize=10)
        ax.set_yticks(range(n_layers + 1))
        ax.set_yticklabels(qry_labels, fontsize=10)
        ax.set_xlabel("Source layer  $h_i$", fontsize=13)
        ax.set_ylabel("Query  $q_l$  (target layer)", fontsize=13)
        ax.set_title(title, fontsize=14)
        for row in range(n_layers + 1):
            for col in range(row + 1):
                val = mat[row, col]
                if not np.isnan(val):
                    txt_color = "white" if val > vmax * 0.6 else "black"
                    ax.text(col, row, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color=txt_color)
        plt.colorbar(im, ax=ax, shrink=0.8, label="Mean softmax weight")

    fig.suptitle(
        r"AttnResMLP Depth-wise Attention Weights  $\alpha_{i \rightarrow l}$"
        "\n(reproduces AttnRes paper Fig.8)",
        fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "mnist_attn_heatmap", out_dir)

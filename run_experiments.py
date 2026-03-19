#!/usr/bin/env python3
"""Run all feature convergence experiments.

Usage:
    python run_experiments.py              # Run all experiments
    python run_experiments.py --exp 1      # Run specific experiment
    python run_experiments.py --exp 1 2    # Run experiments 1 and 2

Experiments:
    1. Cone effect: cosine similarity vs depth/activation
    2. Residual connection comparison: MLP vs ResidualMLP vs AttnResMLP
    3. Layer-wise relative residual (rank collapse analysis)
    4. Random seed sensitivity (different cones)
    5. Transformer rank collapse vs depth  (Noci et al., 2022)
    6. Normalization mode comparison: none/post_ln/pre_ln/rmsnorm
    7. Residual scaling: α=1 vs α=1/√L
    8. Init variance sweep: effect of weight init scale on rank collapse
"""

import argparse
import time
import numpy as np
from pathlib import Path

from src.experiments import (
    exp1_cone_effect,
    exp2_residual_comparison,
    exp3_layerwise_residual,
    exp4_seed_sensitivity,
    exp5_transformer_rank_collapse,
    exp6_norm_comparison,
    exp7_residual_scaling,
    exp8_init_variance,
)
from src.plotting import (
    plot_cone_effect,
    plot_residual_comparison,
    plot_layerwise_residual,
    plot_seed_cones,
    plot_cosine_histogram,
    plot_transformer_rank_collapse,
    plot_norm_comparison,
    plot_residual_scaling,
    plot_init_variance,
)
from src.models import MLP
from src.metrics import cosine_similarity_stats


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def run_exp1(out_dir: str = "figures"):
    """Experiment 1: Cone Effect Verification."""
    separator("Exp 1: Cone Effect — Cosine Similarity vs Depth")
    print("  Reproduces Fig.2(b) from 'Mind the Gap' (Liang et al., 2022)")
    print("  Testing: MLP with different activations, varying depth\n")

    t0 = time.time()
    results = exp1_cone_effect(
        d=512, max_layers=25, n_samples=1000, seed=42,
    )
    plot_cone_effect(results, out_dir)

    # Print summary
    for act, data in results.items():
        final_sim = data[-1][1]
        print(f"  {act:25s}  depth=25 → avg cos_sim = {final_sim:.4f}")

    print(f"\n  Time: {time.time() - t0:.1f}s")


def run_exp2(out_dir: str = "figures"):
    """Experiment 2: Residual Connection Comparison."""
    separator("Exp 2: Residual Connection Comparison")
    print("  Comparing: Plain MLP / ResidualMLP / AttnResMLP")
    print("  Metrics: cosine similarity, effective rank, relative residual\n")

    t0 = time.time()
    results = exp2_residual_comparison(
        d_input=1024, d_hidden=1024, max_layers=20, n_samples=300, seed=42,
    )
    plot_residual_comparison(results, out_dir)

    # Print summary at final depth
    for label, data in results.items():
        cs = data["cos_sim"][-1]
        er = data["eff_rank"][-1]
        rr = data["rel_residual"][-1]
        print(f"  {label:40s}  cos_sim={cs:.3f}  eff_rank={er:.1f}  rel_res={rr:.4f}")

    print(f"\n  Time: {time.time() - t0:.1f}s")


def run_exp3(out_dir: str = "figures"):
    """Experiment 3: Layer-wise Relative Residual."""
    separator("Exp 3: Layer-wise Relative Residual")
    print("  Reproduces Fig.2 from 'Attention is not all you need' (Dong et al.)")
    print("  Measures rank collapse layer-by-layer\n")

    t0 = time.time()
    results = exp3_layerwise_residual(
        d_input=256, d_hidden=256, n_layers=16, n_samples=200, seed=42,
    )
    plot_layerwise_residual(results, out_dir)

    for label, rr in results.items():
        print(f"  {label:25s}  final rel_residual = {rr[-1]:.4f}")

    print(f"\n  Time: {time.time() - t0:.1f}s")


def run_exp4(out_dir: str = "figures"):
    """Experiment 4: Different Seeds → Different Cones."""
    separator("Exp 4: Random Seed Sensitivity")
    print("  Reproduces Fig.2(c) from 'Mind the Gap'")
    print("  Shows different initializations create different cones\n")

    t0 = time.time()
    data = exp4_seed_sensitivity(
        d_input=512, d_hidden=512, n_layers=5, n_samples=500, n_seeds=10,
    )
    plot_seed_cones(data, out_dir)

    # Also plot cosine similarity histogram for one seed
    feats = data["features"][0]
    plot_cosine_histogram(
        feats,
        title="Pairwise Cosine Similarity (Random Init)",
        label="seed=0, 5-layer MLP",
        out_dir=out_dir,
        filename="exp4_cosine_hist",
    )

    # Print cosine stats per seed
    for i, (s, f) in enumerate(zip(data["seeds"], data["features"])):
        stats = cosine_similarity_stats(f)
        if i < 5:  # Print first 5
            print(f"  seed={s}  avg_cos_sim={stats['mean']:.4f}  "
                  f"min={stats['min']:.4f}  max={stats['max']:.4f}")
    if len(data["seeds"]) > 5:
        print(f"  ... ({len(data['seeds']) - 5} more seeds omitted)")

    print(f"\n  Time: {time.time() - t0:.1f}s")


def run_exp5(out_dir: str = "figures"):
    """Experiment 5: Transformer Rank Collapse vs Depth."""
    separator("Exp 5: Transformer Rank Collapse vs Depth")
    print("  Reference: Noci et al. (2022), arXiv:2206.03126")
    print("  Model: TransformerEncoder (pre_ln), Xavier init\n")

    t0 = time.time()
    results = exp5_transformer_rank_collapse(
        d_model=256, max_layers=24, n_samples=300, norm_mode="pre_ln", seed=42,
    )
    plot_transformer_rank_collapse(results, out_dir)

    final = results["n_layers"][-1]
    print(f"  depth={final}  cos_sim={results['cos_sim'][-1]:.4f}  "
          f"eff_rank={results['eff_rank'][-1]:.1f}  "
          f"rel_residual={results['rel_residual'][-1]:.4f}")
    print(f"\n  Time: {time.time() - t0:.1f}s")


def run_exp6(out_dir: str = "figures"):
    """Experiment 6: Normalization Mode Comparison."""
    separator("Exp 6: Normalization Mode Comparison")
    print("  Comparing: none / post_ln / pre_ln / rmsnorm")
    print("  Metrics: cosine similarity, effective rank, relative residual\n")

    t0 = time.time()
    results = exp6_norm_comparison(
        d_model=256, max_layers=16, n_samples=300, seed=42,
    )
    plot_norm_comparison(results, out_dir)

    for norm, data in results.items():
        print(f"  {norm:8s}  depth=16  cos_sim={data['cos_sim'][-1]:.4f}  "
              f"eff_rank={data['eff_rank'][-1]:.1f}  "
              f"rel_residual={data['rel_residual'][-1]:.4f}")
    print(f"\n  Time: {time.time() - t0:.1f}s")


def run_exp7(out_dir: str = "figures"):
    """Experiment 7: Residual Scaling α=1 vs α=1/√L."""
    separator("Exp 7: Residual Scaling — α=1 vs α=1/√L")
    print("  Reference: Noci et al. (2022) §4, depth-scaled residuals")
    print("  Metric: Pearson corr(input pairwise sim, output pairwise sim)\n")

    t0 = time.time()
    results = exp7_residual_scaling(
        d_model=256,
        depths=[2, 4, 8, 12, 16, 24, 32],
        n_samples=300,
        norm_mode="none",
        seed=42,
    )
    plot_residual_scaling(results, out_dir)

    for label in ("alpha_1", "alpha_inv_sqrt"):
        display = "α=1    " if label == "alpha_1" else "α=1/√L "
        corrs = results[label]["correlation"]
        for L, c in zip(results["depths"], corrs):
            pass  # iterate to get last
        print(f"  {display}  L=32  corr={corrs[-1]:.4f}  "
              f"cos_sim={results[label]['cos_sim'][-1]:.4f}")
    print(f"\n  Time: {time.time() - t0:.1f}s")


def run_exp8(out_dir: str = "figures"):
    """Experiment 8: Init Variance Sweep."""
    separator("Exp 8: Init Variance Sweep")
    print("  Reference: Noci et al. (2022) Eq.17-18")
    print("  Sweeping init variance scale m ∈ {0.1, 0.5, 1, 2, 5, 10} × 1/d")
    print("  Fixed: d=1024, n_layers=15, n_samples=300\n")

    t0 = time.time()
    results = exp8_init_variance(
        d=1024, n_layers=15, n_samples=300,
        scale_mults=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        seed=42,
    )
    plot_init_variance(results, out_dir)

    scale_mults = results["scale_mults"]
    for label in ("MLP (no residual)", "ResidualMLP (h+f(h))", "AttnResMLP (attention residual)"):
        if label not in results:
            continue
        data = results[label]
        xavier_idx = scale_mults.index(1.0)
        he_idx = scale_mults.index(2.0)
        print(f"  {label:40s}  Xavier(m=1): cos_sim={data['cos_sim'][xavier_idx]:.3f} "
              f"eff_rank={data['eff_rank'][xavier_idx]:.1f}  "
              f"He(m=2): cos_sim={data['cos_sim'][he_idx]:.3f} "
              f"eff_rank={data['eff_rank'][he_idx]:.1f}")

    print(f"\n  Time: {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", nargs="*", type=int, default=None,
                        help="Which experiments to run (1-7). Default: all.")
    parser.add_argument("--out-dir", default="figures",
                        help="Output directory for figures.")
    args = parser.parse_args()

    exps = args.exp if args.exp else [1, 2, 3, 4, 5, 6, 7, 8]
    out_dir = args.out_dir

    print("Feature Convergence & Residual Connection Experiments")
    print("=" * 60)
    print(f"Output directory: {out_dir}/")
    print(f"Running experiments: {exps}")

    runners = {
        1: run_exp1, 2: run_exp2, 3: run_exp3, 4: run_exp4,
        5: run_exp5, 6: run_exp6, 7: run_exp7, 8: run_exp8,
    }
    for e in exps:
        if e in runners:
            runners[e](out_dir)
        else:
            print(f"  Unknown experiment: {e}")

    separator("Done!")
    print(f"  All figures saved to {out_dir}/\n")


if __name__ == "__main__":
    main()

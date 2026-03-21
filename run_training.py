#!/usr/bin/env python3
"""MNIST training experiments: PlainMLP vs ResidualMLP(α=0.1) vs AttnResMLP.

Trains all three models on MNIST (784→256×8→10) and tracks:
  - Train loss / test accuracy over epochs
  - Cosine similarity and effective rank of final-layer features

For AttnResMLP, also visualises the depth-wise softmax attention weight
matrix before and after training (AttnRes paper Fig.8 style).

Usage:
    python run_training.py                       # defaults: 15 epochs, CPU
    python run_training.py --epochs 20
    python run_training.py --device cuda
    python run_training.py --out-dir results/figures
"""

import argparse
import time

import numpy as np
import torch
from pathlib import Path

from src.training import (
    PlainMLP,
    ResidualMLP,
    AttnResMLP,
    get_mnist_loaders,
    train_model,
    extract_attn_matrix,
)
from src.plotting import plot_training_curves, plot_attn_heatmap


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--epochs",  type=int, default=15,
                        help="Number of training epochs (default: 15)")
    parser.add_argument("--device",  default="cpu",
                        help="PyTorch device (default: cpu)")
    parser.add_argument("--out-dir", default="figures",
                        help="Output directory for figures (default: figures/)")
    args = parser.parse_args()

    device  = torch.device(args.device)
    out_dir = args.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("  MNIST Training: PlainMLP / ResidualMLP(α=0.1) / AttnResMLP")
    print("=" * 62)
    print(f"  epochs={args.epochs}  |  device={device}  |  out={out_dir}/\n")

    # ── Data ──────────────────────────────────────────────────────────────
    print("  Loading MNIST (downloading if needed)...")
    train_loader, test_loader = get_mnist_loaders(batch_size=256, data_dir="data")
    print(f"  Train: {len(train_loader.dataset):,}  |  Test: {len(test_loader.dataset):,}\n")

    # ── Build models ──────────────────────────────────────────────────────
    model_kwargs = dict(d_input=784, d_hidden=256, n_layers=8, n_classes=10)
    plain    = PlainMLP(**model_kwargs).to(device)
    residual = ResidualMLP(**model_kwargs, alpha=0.1).to(device)
    attn     = AttnResMLP(**model_kwargs).to(device)

    # ── Attention matrix BEFORE training (queries=0 → uniform) ───────────
    print("  Extracting attention weights before training...")
    attn_before = extract_attn_matrix(attn, test_loader, device, n_batches=5)
    print("  Done.\n")

    # ── Train ─────────────────────────────────────────────────────────────
    histories = {}
    for name, model in [
        ("PlainMLP",            plain),
        ("ResidualMLP (α=0.1)", residual),
        ("AttnResMLP",          attn),
    ]:
        t0 = time.time()
        histories[name] = train_model(
            model, train_loader, test_loader,
            n_epochs=args.epochs, lr=1e-3, device=device, name=name,
        )
        elapsed = time.time() - t0
        print(f"  Finished {name} in {elapsed:.1f}s")

    # ── Attention matrix AFTER training ───────────────────────────────────
    print("\n  Extracting attention weights after training...")
    attn_after = extract_attn_matrix(attn, test_loader, device, n_batches=10)
    print("  Done.\n")

    # ── Plots ─────────────────────────────────────────────────────────────
    print(f"  Saving figures to {out_dir}/...")
    plot_training_curves(histories, out_dir)
    plot_attn_heatmap(attn_before, attn_after, n_layers=8, out_dir=out_dir)
    print(f"  Saved: {out_dir}/mnist_training_curves.png")
    print(f"  Saved: {out_dir}/mnist_attn_heatmap.png")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n  ── Final results (epoch {args.epochs}) ──")
    print(f"  {'Model':30s}  {'Acc':>6}  {'CosSim':>7}  {'EffRank':>8}")
    print(f"  {'─'*30}  {'─'*6}  {'─'*7}  {'─'*8}")
    for name, h in histories.items():
        print(f"  {name:30s}  {h['test_acc'][-1]:6.4f}  "
              f"{h['cos_sim'][-1]:7.4f}  {h['eff_rank'][-1]:8.1f}")

    # Print learned attention pattern summary
    n_layers = 8
    print(f"\n  ── AttnResMLP attention shift ──")
    print(f"  {'Query':>8}  {'before (top source)':>22}  {'after (top source)':>22}")
    print(f"  {'─'*8}  {'─'*22}  {'─'*22}")
    qry_labels = [f"q{l}" for l in range(n_layers)] + ["q_out"]
    for l in range(n_layers + 1):
        before_row = attn_before[l, :l + 1]
        after_row  = attn_after[l,  :l + 1]
        if len(before_row) < 2:
            continue
        top_before = int(np.argmax(before_row))
        top_after  = int(np.argmax(after_row))
        print(f"  {qry_labels[l]:>8}  "
              f"h{top_before} ({before_row[top_before]:.3f})            "
              f"  h{top_after} ({after_row[top_after]:.3f})")

    print("\n  Done!\n")


if __name__ == "__main__":
    main()

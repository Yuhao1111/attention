"""PyTorch training framework for MNIST classification.

Trains and compares three architectures on MNIST:
  - PlainMLP:          standard feedforward, baseline cone effect
  - ResidualMLP(α=0.1): skip connections with h = h + α·f(h)
  - AttnResMLP:        depth-wise softmax attention over all previous layers

Tracks per epoch: train loss, test accuracy, cosine similarity, effective rank.
After training, extracts the AttnResMLP attention weight matrix (source → target
layer) for visualization, reproducing the style of AttnRes paper Fig.8.

Reference:
    Kimi Team (2026), "Attention Residuals", arXiv:2603.15031
    Noci et al. (2022), "Signal Propagation in Transformers", arXiv:2206.03126
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Dict, List, Tuple, Any

from .metrics import cosine_similarity_stats, effective_rank


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rms_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """RMSNorm along last dimension."""
    return x / x.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class PlainMLP(nn.Module):
    """Standard feedforward network: expected to exhibit cone effect."""

    def __init__(self, d_input: int = 784, d_hidden: int = 256,
                 n_layers: int = 8, n_classes: int = 10):
        super().__init__()
        self.n_layers = n_layers
        self.proj_in  = nn.Linear(d_input, d_hidden)
        self.layers   = nn.ModuleList(
            [nn.Linear(d_hidden, d_hidden) for _ in range(n_layers)]
        )
        self.head = nn.Linear(d_hidden, n_classes)
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        h = F.relu(self.proj_in(x))
        feats = [h] if return_features else None
        for layer in self.layers:
            h = F.relu(layer(h))
            if return_features:
                feats.append(h)
        logits = self.head(h)
        return (logits, feats) if return_features else logits


class ResidualMLP(nn.Module):
    """MLP with skip connections: h = h + alpha * f(h)."""

    def __init__(self, d_input: int = 784, d_hidden: int = 256,
                 n_layers: int = 8, n_classes: int = 10, alpha: float = 0.1):
        super().__init__()
        self.n_layers = n_layers
        self.alpha    = alpha
        self.proj_in  = nn.Linear(d_input, d_hidden)
        self.layers   = nn.ModuleList(
            [nn.Linear(d_hidden, d_hidden) for _ in range(n_layers)]
        )
        self.head = nn.Linear(d_hidden, n_classes)
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        h = F.relu(self.proj_in(x))
        feats = [h] if return_features else None
        for layer in self.layers:
            h = h + self.alpha * F.relu(layer(h))
            if return_features:
                feats.append(h)
        logits = self.head(h)
        return (logits, feats) if return_features else logits


class AttnResMLP(nn.Module):
    """MLP with learned depth-wise attention residuals.

    For each layer l, a learned pseudo-query q_l computes softmax attention
    over all preceding source representations [h_0, ..., h_{l-1}]:

        alpha_{i→l} = softmax_i( q_l · RMSNorm(h_i) )
        h_input_l   = Σ_i  alpha_{i→l} · h_i
        h_l         = ReLU( W_l · h_input_l + b_l )

    A final attention aggregation over all n_layers+1 representations feeds
    the classification head.

    Queries are initialised to zero → uniform attention at initialisation.
    They are nn.Parameter, so they are learned by the optimiser.

    Attention weight matrix (for visualisation):
        Shape: (n_layers+1, n_layers+1), lower triangular.
        Row l, col i = mean softmax weight of source i at query l.
        Row 0..n_layers-1 correspond to transform queries.
        Row n_layers corresponds to the output (head-input) query.

    Reference: Kimi Team (2026), "Attention Residuals".
    """

    def __init__(self, d_input: int = 784, d_hidden: int = 256,
                 n_layers: int = 8, n_classes: int = 10):
        super().__init__()
        self.n_layers = n_layers
        self.d_hidden = d_hidden
        self.proj_in  = nn.Linear(d_input, d_hidden)
        self.layers   = nn.ModuleList(
            [nn.Linear(d_hidden, d_hidden) for _ in range(n_layers)]
        )
        # n_layers transform queries + 1 output query = n_layers+1 total
        # Zero init → uniform softmax attention at the start of training
        self.queries = nn.Parameter(torch.zeros(n_layers + 1, d_hidden))
        self.head    = nn.Linear(d_hidden, n_classes)
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        # queries: left at zero (set in __init__ via nn.Parameter)

    def _aggregate(
        self,
        query: torch.Tensor,           # (d,)
        sources: List[torch.Tensor],   # list of S tensors, each (N, d)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Softmax attention over source representations.

        Returns:
            h_agg:   (N, d) weighted combination of sources
            weights: (S, N) softmax attention weights
        """
        V = torch.stack(sources, dim=0)                        # (S, N, d)
        K = _rms_norm(V)                                       # (S, N, d)
        logits  = torch.einsum("d, s n d -> s n", query, K)   # (S, N)
        weights = F.softmax(logits, dim=0)                     # (S, N)
        h_agg   = torch.einsum("s n, s n d -> n d", weights, V)  # (N, d)
        return h_agg, weights

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_attn: bool = False,
    ):
        """Forward pass.

        Returns:
            logits                                 always
            features  (List of (N,d) tensors)      if return_features=True
            attn_matrix (np.ndarray, L+1 × L+1)   if return_attn=True

        attn_matrix[l, i] = mean softmax weight of source i at query l.
        Entry is 0 for i > l (source not yet available).
        """
        h0 = F.relu(self.proj_in(x))
        sources: List[torch.Tensor] = [h0]
        feats = [h0] if return_features else None
        attn_rows: List[torch.Tensor] = []

        for l, layer in enumerate(self.layers):
            h_agg, w = self._aggregate(self.queries[l], sources)
            if return_attn:
                # w: (l+1, N) → mean over batch → (l+1,) → pad to n_layers+1
                row = F.pad(w.detach().mean(dim=1),
                            (0, self.n_layers + 1 - (l + 1)))
                attn_rows.append(row)
            h = F.relu(layer(h_agg))
            sources.append(h)
            if return_features:
                feats.append(h)

        # Final aggregation for classification (sources now has n_layers+1 items)
        h_out, w_out = self._aggregate(self.queries[self.n_layers], sources)
        if return_attn:
            attn_rows.append(w_out.detach().mean(dim=1))   # already n_layers+1 wide
        logits = self.head(h_out)

        if not return_features and not return_attn:
            return logits
        out = [logits]
        if return_features:
            out.append(feats)
        if return_attn:
            # (n_layers+1, n_layers+1) lower-triangular weight matrix
            out.append(torch.stack(attn_rows).cpu().numpy())
        return tuple(out)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_mnist_loaders(
    batch_size: int = 256,
    data_dir: str = "data",
) -> Tuple[DataLoader, DataLoader]:
    """Return MNIST train/test DataLoaders (downloads on first run)."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda t: t.view(-1)),   # 28×28 → 784
    ])
    kw = dict(download=True, transform=transform)
    train_ds = datasets.MNIST(data_dir, train=True,  **kw)
    test_ds  = datasets.MNIST(data_dir, train=False, **kw)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / n


def _eval(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Return (test_loss, test_accuracy)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item() * x.size(0)
            correct    += (logits.argmax(1) == y).sum().item()
            n          += x.size(0)
    return total_loss / n, correct / n


def _feature_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_samples: int = 2000,
) -> Tuple[float, float]:
    """Compute cosine similarity and effective rank of final-layer features.

    Returns:
        (mean_cosine_similarity, effective_rank)
    """
    model.eval()
    collected, n = [], 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            result = model(x, return_features=True)
            # result = (logits, features_list); take last feature
            collected.append(result[1][-1].cpu().numpy())
            n += x.size(0)
            if n >= n_samples:
                break
    arr = np.vstack(collected)[:n_samples]
    return cosine_similarity_stats(arr)["mean"], effective_rank(arr)


def extract_attn_matrix(
    model: AttnResMLP,
    loader: DataLoader,
    device: torch.device,
    n_batches: int = 10,
) -> np.ndarray:
    """Average attention weight matrix over multiple batches.

    Returns:
        np.ndarray of shape (n_layers+1, n_layers+1), lower triangular.
        Row l, col i = mean softmax weight of source i at query l.
    """
    model.eval()
    mats = []
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= n_batches:
                break
            _, attn = model(x.to(device), return_attn=True)
            mats.append(attn)
    return np.stack(mats).mean(axis=0)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    n_epochs: int = 15,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    name: str = "",
) -> Dict[str, Any]:
    """Train model and collect per-epoch metrics.

    Returns history dict with keys:
        train_loss, test_loss, test_acc, cos_sim, eff_rank
    each a list of length n_epochs.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: Dict[str, list] = {
        k: [] for k in ("train_loss", "test_loss", "test_acc", "cos_sim", "eff_rank")
    }

    label = name or model.__class__.__name__
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  ── {label}  ({n_params:,} params) ──")
    print(f"  {'Ep':>3}  {'TrLoss':>8}  {'TsLoss':>8}  "
          f"{'Acc':>6}  {'CosSim':>7}  {'EffRank':>8}")
    print(f"  {'─'*3}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*7}  {'─'*8}")

    for epoch in range(1, n_epochs + 1):
        tr_loss = _train_epoch(model, train_loader, optimizer, device)
        ts_loss, acc = _eval(model, test_loader, device)
        cos, rank = _feature_metrics(model, test_loader, device)

        history["train_loss"].append(tr_loss)
        history["test_loss"].append(ts_loss)
        history["test_acc"].append(acc)
        history["cos_sim"].append(cos)
        history["eff_rank"].append(rank)

        print(f"  {epoch:3d}  {tr_loss:8.4f}  {ts_loss:8.4f}  "
              f"{acc:6.4f}  {cos:7.4f}  {rank:8.1f}")

    return history

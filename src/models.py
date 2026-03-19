"""Neural network models for feature convergence experiments.

Implements forward-only models using NumPy to study:
- Cone effect in plain MLPs
- Rank collapse mitigation via residual connections
- Attention-based residual aggregation (AttnRes)

All models support layer-wise feature extraction for analysis.
"""

import numpy as np
from typing import List, Tuple, Optional, Callable


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(x > 0, x, alpha * x)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def identity(x: np.ndarray) -> np.ndarray:
    return x


ACTIVATIONS = {
    "relu": relu,
    "sigmoid": sigmoid,
    "leaky_relu": leaky_relu,
    "tanh": tanh,
    "linear": identity,
}


# ---------------------------------------------------------------------------
# Weight initialization
# ---------------------------------------------------------------------------

def init_weights(
    d_in: int,
    d_out: int,
    rng: np.random.Generator,
    scale: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Initialize weight matrix and bias vector.

    Default scale follows the "Mind the Gap" paper setup:
        W ~ N(0, 1/d_out),  b ~ N(0, 1/d_out)
    """
    if scale is None:
        scale = 1.0 / d_out
    W = rng.normal(0, np.sqrt(scale), size=(d_out, d_in))
    b = rng.normal(0, np.sqrt(scale), size=(d_out,))
    return W, b


# ---------------------------------------------------------------------------
# Plain MLP  (no residual)
# ---------------------------------------------------------------------------

class MLP:
    """Multi-layer perceptron without residual connections.

    This is the baseline model expected to exhibit the cone effect:
    feature cosine similarity increases rapidly with depth.
    """

    def __init__(
        self,
        d_input: int,
        d_hidden: int,
        n_layers: int,
        activation: str = "relu",
        seed: int = 0,
    ):
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.activation_name = activation
        self.act_fn = ACTIVATIONS[activation]
        self.seed = seed

        rng = np.random.default_rng(seed)
        self.layers: List[Tuple[np.ndarray, np.ndarray]] = []

        # First layer: d_input -> d_hidden
        self.layers.append(init_weights(d_input, d_hidden, rng))
        # Hidden layers: d_hidden -> d_hidden
        for _ in range(n_layers - 1):
            self.layers.append(init_weights(d_hidden, d_hidden, rng))

    def forward(self, X: np.ndarray, return_intermediates: bool = False):
        """Forward pass.

        Args:
            X: Input array of shape (N, d_input).
            return_intermediates: If True, return features at every layer.

        Returns:
            Output features, or list of features at each layer.
        """
        intermediates = [X] if return_intermediates else None
        h = X

        for i, (W, b) in enumerate(self.layers):
            h = h @ W.T + b  # Linear
            h = self.act_fn(h)  # Nonlinearity
            if return_intermediates:
                intermediates.append(h)

        return intermediates if return_intermediates else h

    def __repr__(self):
        return (
            f"MLP(d_in={self.d_input}, d_hid={self.d_hidden}, "
            f"layers={self.n_layers}, act={self.activation_name}, "
            f"seed={self.seed})"
        )


# ---------------------------------------------------------------------------
# Residual MLP  (standard additive residual: h = h + f(h))
# ---------------------------------------------------------------------------

class ResidualMLP:
    """MLP with standard residual (skip) connections.

    Update rule:  h_l = h_{l-1} + f_{l-1}(h_{l-1})

    Expected to mitigate the cone effect / rank collapse.
    See: He et al. (2015), Dong et al. (2021) §3.1
    """

    def __init__(
        self,
        d_input: int,
        d_hidden: int,
        n_layers: int,
        activation: str = "relu",
        seed: int = 0,
    ):
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.activation_name = activation
        self.act_fn = ACTIVATIONS[activation]
        self.seed = seed

        rng = np.random.default_rng(seed)
        self.layers: List[Tuple[np.ndarray, np.ndarray]] = []

        # Project input to hidden dim
        self.proj_in = init_weights(d_input, d_hidden, rng)

        # Residual layers: d_hidden -> d_hidden
        for _ in range(n_layers):
            self.layers.append(init_weights(d_hidden, d_hidden, rng))

    def forward(self, X: np.ndarray, return_intermediates: bool = False):
        """Forward pass with residual connections."""
        # Project to hidden space
        W0, b0 = self.proj_in
        h = X @ W0.T + b0

        intermediates = [h] if return_intermediates else None

        for W, b in self.layers:
            residual = h
            h = h @ W.T + b
            h = self.act_fn(h)
            h = residual + h  # Skip connection
            if return_intermediates:
                intermediates.append(h)

        return intermediates if return_intermediates else h

    def __repr__(self):
        return (
            f"ResidualMLP(d_in={self.d_input}, d_hid={self.d_hidden}, "
            f"layers={self.n_layers}, act={self.activation_name}, "
            f"seed={self.seed})"
        )


# ---------------------------------------------------------------------------
# Attention Residuals MLP  (AttnRes-style depth-wise attention)
# ---------------------------------------------------------------------------

def rms_norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """RMSNorm along the last axis."""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return x / rms


class AttnResMLP:
    """MLP with Attention Residuals (simplified version).

    Instead of h_l = sum_i v_i (standard residual = uniform weights),
    we compute h_l = sum_i alpha_{i->l} * v_i,
    where alpha are softmax attention weights from a learned pseudo-query.

    This implements the core idea from "Attention Residuals" (Kimi Team, 2026).
    """

    def __init__(
        self,
        d_input: int,
        d_hidden: int,
        n_layers: int,
        activation: str = "relu",
        seed: int = 0,
    ):
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.activation_name = activation
        self.act_fn = ACTIVATIONS[activation]
        self.seed = seed

        rng = np.random.default_rng(seed)

        # Input projection
        self.proj_in = init_weights(d_input, d_hidden, rng)

        # Transform layers
        self.layers: List[Tuple[np.ndarray, np.ndarray]] = []
        for _ in range(n_layers):
            self.layers.append(init_weights(d_hidden, d_hidden, rng))

        # Pseudo-query vectors for AttnRes (one per layer, initialized to zero)
        # Zero init => uniform attention at start, as recommended in the paper
        self.queries = np.zeros((n_layers, d_hidden))

    def _attn_aggregate(
        self,
        query: np.ndarray,
        sources: List[np.ndarray],
    ) -> np.ndarray:
        """Compute softmax attention over source representations.

        Args:
            query:   (d_hidden,) — learned pseudo-query for this layer.
            sources: list of (N, d_hidden) — all preceding layer outputs.

        Returns:
            (N, d_hidden) — weighted combination of sources.
        """
        # Stack sources: (num_sources, N, d_hidden)
        V = np.stack(sources, axis=0)
        K = rms_norm(V)  # Normalize keys

        # Compute logits: query dot each key -> (num_sources, N)
        logits = np.einsum("d, s n d -> s n", query, K)

        # Softmax over sources dimension
        logits_max = logits.max(axis=0, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        weights = exp_logits / exp_logits.sum(axis=0, keepdims=True)

        # Weighted sum: (N, d_hidden)
        h = np.einsum("s n, s n d -> n d", weights, V)
        return h

    def forward(self, X: np.ndarray, return_intermediates: bool = False):
        """Forward pass with attention residuals."""
        W0, b0 = self.proj_in
        h0 = X @ W0.T + b0

        # Sources: all previous layer outputs (v_0 = h0, v_i = f_i(h_i))
        sources = [h0]
        intermediates = [h0] if return_intermediates else None

        for i, (W, b) in enumerate(self.layers):
            # Aggregate previous outputs via attention
            h = self._attn_aggregate(self.queries[i], sources)

            # Apply transform
            out = h @ W.T + b
            out = self.act_fn(out)

            sources.append(out)
            if return_intermediates:
                intermediates.append(
                    self._attn_aggregate(self.queries[min(i + 1, self.n_layers - 1)], sources)
                )

        # Final output = attention aggregate over all sources
        final = self._attn_aggregate(self.queries[-1], sources)
        if return_intermediates:
            intermediates[-1] = final
        return intermediates if return_intermediates else final

    def __repr__(self):
        return (
            f"AttnResMLP(d_in={self.d_input}, d_hid={self.d_hidden}, "
            f"layers={self.n_layers}, act={self.activation_name}, "
            f"seed={self.seed})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "mlp": MLP,
    "residual_mlp": ResidualMLP,
    "attnres_mlp": AttnResMLP,
}


def build_model(name: str, **kwargs):
    """Build a model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)

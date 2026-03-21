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
        init_scale: Optional[float] = None,
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
        self.layers.append(init_weights(d_input, d_hidden, rng, scale=init_scale))
        # Hidden layers: d_hidden -> d_hidden
        for _ in range(n_layers - 1):
            self.layers.append(init_weights(d_hidden, d_hidden, rng, scale=init_scale))

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

    Update rule:  h_l = h_{l-1} + alpha * f_{l-1}(h_{l-1})

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
        init_scale: Optional[float] = None,
        alpha: float = 1.0,
    ):
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.activation_name = activation
        self.act_fn = ACTIVATIONS[activation]
        self.seed = seed
        self.alpha = alpha  # Residual scaling: h = alpha * h + f(h)

        rng = np.random.default_rng(seed)
        self.layers: List[Tuple[np.ndarray, np.ndarray]] = []

        # Project input to hidden dim
        self.proj_in = init_weights(d_input, d_hidden, rng, scale=init_scale)

        # Residual layers: d_hidden -> d_hidden
        for _ in range(n_layers):
            self.layers.append(init_weights(d_hidden, d_hidden, rng, scale=init_scale))

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
            h = residual + self.alpha * h  # h = h_old + alpha * f(h_old)
            if return_intermediates:
                intermediates.append(h)

        return intermediates if return_intermediates else h

    def __repr__(self):
        return (
            f"ResidualMLP(d_in={self.d_input}, d_hid={self.d_hidden}, "
            f"layers={self.n_layers}, act={self.activation_name}, "
            f"alpha={self.alpha:.3f}, seed={self.seed})"
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
        init_scale: Optional[float] = None,
    ):
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.activation_name = activation
        self.act_fn = ACTIVATIONS[activation]
        self.seed = seed

        rng = np.random.default_rng(seed)

        # Input projection
        self.proj_in = init_weights(d_input, d_hidden, rng, scale=init_scale)

        # Transform layers
        self.layers: List[Tuple[np.ndarray, np.ndarray]] = []
        for _ in range(n_layers):
            self.layers.append(init_weights(d_hidden, d_hidden, rng, scale=init_scale))

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
# Pure Self-Attention  (no residual, no FFN)
# ---------------------------------------------------------------------------

class PureSelfAttention:
    """Pure self-attention layers without any residual connections.

    Each layer computes:
        h_{l+1} = softmax(h_l W_Q (h_l W_K)^T / sqrt(d)) h_l W_V W_O

    No skip connections are applied.  This is the degenerate architecture
    studied by Dong et al. (2021, ICML), who prove that the rank of h_l
    collapses doubly-exponentially with depth:
        rank(h_l) ≤ n / 2^l  (approximately)

    All weight matrices are Xavier-initialized: W ~ N(0, 1/d).

    The model accepts the same forward(X, return_intermediates=True)
    interface as MLP / ResidualMLP / AttnResMLP for drop-in comparison.
    """

    def __init__(
        self,
        d_input: int,
        d_hidden: int,
        n_layers: int,
        activation: str = "relu",   # accepted for API compatibility, unused
        seed: int = 0,
        init_scale: Optional[float] = None,
    ):
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.seed = seed

        rng = np.random.default_rng(seed)
        scale = init_scale if init_scale is not None else 1.0 / d_hidden

        # Input projection (d_input -> d_hidden)
        self.W_in = rng.normal(0, np.sqrt(1.0 / d_input), (d_hidden, d_input))

        # Per-layer attention matrices
        self.blocks: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for _ in range(n_layers):
            W_Q = rng.normal(0, np.sqrt(scale), (d_hidden, d_hidden))
            W_K = rng.normal(0, np.sqrt(scale), (d_hidden, d_hidden))
            W_V = rng.normal(0, np.sqrt(scale), (d_hidden, d_hidden))
            W_O = rng.normal(0, np.sqrt(scale), (d_hidden, d_hidden))
            self.blocks.append((W_Q, W_K, W_V, W_O))

    def _attn(
        self,
        X: np.ndarray,
        W_Q: np.ndarray,
        W_K: np.ndarray,
        W_V: np.ndarray,
        W_O: np.ndarray,
    ) -> np.ndarray:
        Q = X @ W_Q.T
        K = X @ W_K.T
        V = X @ W_V.T
        scale = np.sqrt(self.d_hidden)
        scores = Q @ K.T / scale                        # (N, N)
        scores -= scores.max(axis=-1, keepdims=True)    # numerical stability
        weights = np.exp(scores)
        weights /= weights.sum(axis=-1, keepdims=True)
        return weights @ V @ W_O.T                      # (N, d_hidden)

    def forward(self, X: np.ndarray, return_intermediates: bool = False):
        """Forward pass: pure self-attention, no residual."""
        h = X @ self.W_in.T                             # (N, d_hidden)
        intermediates = [h] if return_intermediates else None

        for W_Q, W_K, W_V, W_O in self.blocks:
            h = self._attn(h, W_Q, W_K, W_V, W_O)     # No skip connection
            if return_intermediates:
                intermediates.append(h)

        return intermediates if return_intermediates else h

    def __repr__(self):
        return (
            f"PureSelfAttention(d_in={self.d_input}, d_hid={self.d_hidden}, "
            f"layers={self.n_layers}, seed={self.seed})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "mlp": MLP,
    "residual_mlp": ResidualMLP,
    "attnres_mlp": AttnResMLP,
    "pure_self_attn": PureSelfAttention,
}


def build_model(name: str, **kwargs):
    """Build a model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)


# ---------------------------------------------------------------------------
# Transformer Block  (self-attention + FFN + residual + normalization)
# ---------------------------------------------------------------------------

def _layer_norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """LayerNorm along the last axis."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


class TransformerBlock:
    """Single Transformer block: self-attention + FFN + residual connections.

    Forward pass depending on norm_mode:

        none:
            Z    = X + α1 * Attn(X)
            out  = Z + α2 * FFN(Z)

        post_ln:
            Z    = LayerNorm(X + α1 * Attn(X))
            out  = LayerNorm(Z + α2 * FFN(Z))

        pre_ln:
            Z    = X + α1 * Attn(LayerNorm(X))
            out  = Z + α2 * FFN(LayerNorm(Z))

        rmsnorm:
            Z    = X + α1 * Attn(RMSNorm(X))
            out  = Z + α2 * FFN(RMSNorm(Z))

    All weights are Xavier-initialized: W ~ N(0, 1/d_model).

    Reference:
        Noci et al., "Signal Propagation in Transformers" (2022).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        norm_mode: str = "pre_ln",
        alpha1: float = 1.0,
        alpha2: float = 1.0,
        seed: int = 0,
    ):
        assert norm_mode in ("none", "post_ln", "pre_ln", "rmsnorm"), (
            f"norm_mode must be one of: none, post_ln, pre_ln, rmsnorm. Got: {norm_mode}"
        )
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        self.norm_mode = norm_mode
        self.alpha1 = alpha1
        self.alpha2 = alpha2

        rng = np.random.default_rng(seed)
        scale_attn = 1.0 / d_model        # Xavier for attention projections
        scale_ff1  = 1.0 / d_model        # Xavier for FFN first layer
        scale_ff2  = 1.0 / self.d_ff      # Xavier for FFN second layer

        # Self-attention projections  (d_model → d_model each)
        self.W_Q = rng.normal(0, np.sqrt(scale_attn), (d_model, d_model))
        self.W_K = rng.normal(0, np.sqrt(scale_attn), (d_model, d_model))
        self.W_V = rng.normal(0, np.sqrt(scale_attn), (d_model, d_model))
        self.W_O = rng.normal(0, np.sqrt(scale_attn), (d_model, d_model))

        # FFN:  d_model → d_ff → d_model
        self.W1 = rng.normal(0, np.sqrt(scale_ff1), (self.d_ff, d_model))
        self.b1 = rng.normal(0, np.sqrt(scale_ff1), (self.d_ff,))
        self.W2 = rng.normal(0, np.sqrt(scale_ff2), (d_model, self.d_ff))
        self.b2 = rng.normal(0, np.sqrt(scale_ff2), (d_model,))

    # ------------------------------------------------------------------
    # Sub-layer helpers
    # ------------------------------------------------------------------

    def _attn(self, X: np.ndarray) -> np.ndarray:
        """Scaled dot-product self-attention: softmax(QK^T / sqrt(d_k)) V W_O."""
        Q = X @ self.W_Q.T                          # (N, d_model)
        K = X @ self.W_K.T
        V = X @ self.W_V.T

        scale = np.sqrt(self.d_model)
        scores = Q @ K.T / scale                    # (N, N)

        # Numerically stable softmax along key dimension
        scores -= scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights /= weights.sum(axis=-1, keepdims=True)

        out = weights @ V                           # (N, d_model)
        return out @ self.W_O.T

    def _ffn(self, X: np.ndarray) -> np.ndarray:
        """FFN: Linear + ReLU + Linear."""
        h = X @ self.W1.T + self.b1
        h = np.maximum(h, 0.0)                     # ReLU
        return h @ self.W2.T + self.b2

    def _norm(self, X: np.ndarray) -> np.ndarray:
        """Apply the block's normalization (used for pre-norm variants)."""
        if self.norm_mode in ("pre_ln", "post_ln"):
            return _layer_norm(X)
        elif self.norm_mode == "rmsnorm":
            return rms_norm(X)
        return X  # none — identity

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Apply one Transformer block.

        Args:
            X: (N, d_model) token/sample feature matrix.

        Returns:
            (N, d_model) updated features.
        """
        if self.norm_mode == "post_ln":
            Z   = _layer_norm(X + self.alpha1 * self._attn(X))
            out = _layer_norm(Z + self.alpha2 * self._ffn(Z))
        elif self.norm_mode in ("pre_ln", "rmsnorm"):
            Z   = X + self.alpha1 * self._attn(self._norm(X))
            out = Z + self.alpha2 * self._ffn(self._norm(Z))
        else:  # none
            Z   = X + self.alpha1 * self._attn(X)
            out = Z + self.alpha2 * self._ffn(Z)
        return out

    def __repr__(self):
        return (
            f"TransformerBlock(d={self.d_model}, d_ff={self.d_ff}, "
            f"norm={self.norm_mode}, α1={self.alpha1:.3f}, α2={self.alpha2:.3f})"
        )


# ---------------------------------------------------------------------------
# Transformer Encoder  (stack of TransformerBlocks)
# ---------------------------------------------------------------------------

class TransformerEncoder:
    """Stack of TransformerBlocks for feature convergence experiments.

    Supports the same forward(X, return_intermediates=True) interface as MLP.

    Args:
        d_model:    Token/feature dimension.
        n_layers:   Number of Transformer blocks.
        d_ff:       FFN hidden dimension (default: 4 × d_model).
        norm_mode:  One of "none", "post_ln", "pre_ln", "rmsnorm".
        alpha1:     Residual scale for the attention branch (default 1.0).
                    Set to 1/√n_layers for the depth-scaled variant from
                    Noci et al. (2022).
        alpha2:     Residual scale for the FFN branch (default 1.0).
        seed:       Master random seed; each block gets a derived seed.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        d_ff: Optional[int] = None,
        norm_mode: str = "pre_ln",
        alpha1: float = 1.0,
        alpha2: float = 1.0,
        seed: int = 0,
    ):
        self.d_model = d_model
        self.n_layers = n_layers
        self.norm_mode = norm_mode
        self.alpha1 = alpha1
        self.alpha2 = alpha2

        rng = np.random.default_rng(seed)
        self.blocks: List[TransformerBlock] = [
            TransformerBlock(
                d_model, d_ff, norm_mode, alpha1, alpha2,
                seed=int(rng.integers(1 << 31)),
            )
            for _ in range(n_layers)
        ]

    def forward(self, X: np.ndarray, return_intermediates: bool = False):
        """Forward pass through all Transformer blocks.

        Args:
            X:                   (N, d_model) input features.
            return_intermediates: If True, return list of features at every
                                  layer (including the input as index 0).

        Returns:
            Output features, or list of features at each layer.
        """
        intermediates = [X] if return_intermediates else None
        h = X
        for block in self.blocks:
            h = block.forward(h)
            if return_intermediates:
                intermediates.append(h)
        return intermediates if return_intermediates else h

    def __repr__(self):
        return (
            f"TransformerEncoder(d={self.d_model}, L={self.n_layers}, "
            f"norm={self.norm_mode}, α1={self.alpha1:.3f}, α2={self.alpha2:.3f})"
        )

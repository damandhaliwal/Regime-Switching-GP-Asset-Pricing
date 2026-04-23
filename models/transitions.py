"""Non-homogeneous logistic transitions (Filardo 1994).

Parameterization. For each source state j ∈ {0, ..., K-1}, the distribution over
the next state k is a softmax on affine combinations of macro covariates z:

    P(s_{t} = k | s_{t-1} = j, z_t) = softmax_k( W[j, k, :] · z_t + b[j, k] )

With K-1 free rows per source state (we fix k = 0 as the reference class:
W[j, 0, :] = 0, b[j, 0] = 0). This matches the K-class multinomial logit.

Linear predictors are clipped to [-CLIP, CLIP] before the softmax so that macro
spikes (e.g. March 2020 VIX) do not produce NaN gradients.

Public API:
- log_transition_matrix(W, b, Z) -> (T-1, K, K) of log P(s_t=k | s_{t-1}=j, z_t)
- fit_logistic_weighted(Z, xi, K, M, ...) -> (W, b) maximizing Σ_t Σ_jk ξ_{t,j,k} log P(k | j, z_t)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

CLIP = 30.0


def log_transition_matrix(W: np.ndarray, b: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Return (T-1, K, K) log-probabilities. Z is (T, M); the t-th transition uses z_{t+1}."""
    T = Z.shape[0]
    K = b.shape[0]
    Z_use = Z[1:]  # transitions at t correspond to z at t+1 (the arrival time)
    # logits[t, j, k] = W[j, k, :] · z_{t+1} + b[j, k]
    logits = np.einsum("jkm,tm->tjk", W, Z_use) + b[None, :, :]
    np.clip(logits, -CLIP, CLIP, out=logits)
    # Reference class: force column 0 to 0 so parameters are identified.
    logits[..., 0] = 0.0
    return logits - logsumexp(logits, axis=2, keepdims=True)


def _pack(W: np.ndarray, b: np.ndarray, K: int, M: int) -> np.ndarray:
    # Only columns k = 1..K-1 are free; k = 0 is reference.
    w_free = W[:, 1:, :].reshape(-1)
    b_free = b[:, 1:].reshape(-1)
    return np.concatenate([w_free, b_free])


def _unpack(v: np.ndarray, K: int, M: int) -> tuple[np.ndarray, np.ndarray]:
    n_w = K * (K - 1) * M
    W = np.zeros((K, K, M))
    b = np.zeros((K, K))
    W[:, 1:, :] = v[:n_w].reshape(K, K - 1, M)
    b[:, 1:] = v[n_w:].reshape(K, K - 1)
    return W, b


def _neg_weighted_ll(v: np.ndarray, K: int, M: int, Z: np.ndarray,
                     xi: np.ndarray, l2: float) -> float:
    W, b = _unpack(v, K, M)
    log_P = log_transition_matrix(W, b, Z)  # (T-1, K, K)
    # xi is (T-1, K, K); sum elementwise: -Σ xi * log_P + l2 regularization.
    total = -(xi * log_P).sum()
    if l2 > 0.0:
        total += 0.5 * l2 * float((v * v).sum())
    return float(total)


def fit_logistic_weighted(Z: np.ndarray, xi: np.ndarray,
                          K: int | None = None, M: int | None = None,
                          l2: float = 1e-4,
                          n_restarts: int = 2,
                          seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Maximum-weighted-likelihood fit of logistic transitions.

    Z: (T, M) macro covariates (use z_t to drive transitions into t; the t-th
       transition pulls z_{t+1} internally via log_transition_matrix).
    xi: (T-1, K, K) soft counts from the E-step.
    l2: Gaussian prior strength on all free params (prevents blow-up when a
        state has near-zero marginal mass).
    """
    if K is None:
        K = xi.shape[1]
    if M is None:
        M = Z.shape[1]

    rng = np.random.default_rng(seed)
    n_params = K * (K - 1) * M + K * (K - 1)
    inits = [np.zeros(n_params)]
    for _ in range(n_restarts):
        inits.append(rng.normal(0.0, 0.1, size=n_params))

    best_v: np.ndarray | None = None
    best_f = np.inf
    for v0 in inits:
        try:
            res = minimize(
                _neg_weighted_ll, v0, args=(K, M, Z, xi, l2),
                method="L-BFGS-B",
                options={"maxiter": 300, "gtol": 1e-6, "ftol": 1e-9},
            )
            if np.isfinite(res.fun) and res.fun < best_f:
                best_f = float(res.fun)
                best_v = res.x
        except Exception:  # pragma: no cover
            continue

    if best_v is None:
        return _unpack(np.zeros(n_params), K, M)
    return _unpack(best_v, K, M)

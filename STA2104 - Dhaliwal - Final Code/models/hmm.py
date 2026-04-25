"""Discrete-state HMM with non-homogeneous transitions.

Log-space forward/backward to stay numerically stable out to T = 300+ months.
Emission log-probabilities are supplied externally; this file knows nothing
about GPs. Transitions are supplied as a (T-1, K, K) array whose [t, j, k]
entry is log P(s_{t+1} = k | s_t = j, z_{t+1}).
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def forward_log(log_pi: np.ndarray,
                log_trans: np.ndarray,
                log_emit: np.ndarray) -> np.ndarray:
    """Return log_alpha of shape (T, K). log_trans has shape (T-1, K, K)."""
    T, K = log_emit.shape
    log_alpha = np.empty((T, K))
    log_alpha[0] = log_pi + log_emit[0]
    for t in range(1, T):
        # log_alpha[t, k] = log_emit[t, k] + logsumexp_j (log_alpha[t-1, j] + log_trans[t-1, j, k])
        tmp = log_alpha[t - 1, :, None] + log_trans[t - 1]  # (K, K)
        log_alpha[t] = log_emit[t] + logsumexp(tmp, axis=0)
    return log_alpha


def backward_log(log_trans: np.ndarray, log_emit: np.ndarray) -> np.ndarray:
    """Return log_beta of shape (T, K)."""
    T, K = log_emit.shape
    log_beta = np.zeros((T, K))
    for t in range(T - 2, -1, -1):
        # log_beta[t, j] = logsumexp_k (log_trans[t, j, k] + log_emit[t+1, k] + log_beta[t+1, k])
        tmp = log_trans[t] + (log_emit[t + 1] + log_beta[t + 1])[None, :]  # (K, K)
        log_beta[t] = logsumexp(tmp, axis=1)
    return log_beta


def posteriors(log_pi: np.ndarray,
               log_trans: np.ndarray,
               log_emit: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Forward-backward. Returns (gamma (T,K), xi (T-1,K,K), log_likelihood)."""
    T, K = log_emit.shape
    log_alpha = forward_log(log_pi, log_trans, log_emit)
    log_beta = backward_log(log_trans, log_emit)
    log_lik = logsumexp(log_alpha[-1])

    log_gamma = log_alpha + log_beta - log_lik
    gamma = np.exp(log_gamma)

    # xi[t, j, k] ∝ α[t, j] + trans[t, j, k] + emit[t+1, k] + β[t+1, k]
    log_xi = (
        log_alpha[:-1, :, None]
        + log_trans
        + (log_emit[1:] + log_beta[1:])[:, None, :]
    )
    # Normalize per t.
    log_xi -= logsumexp(log_xi, axis=(1, 2), keepdims=True)
    xi = np.exp(log_xi)
    return gamma, xi, float(log_lik)


def viterbi(log_pi: np.ndarray,
            log_trans: np.ndarray,
            log_emit: np.ndarray) -> np.ndarray:
    """Most-likely state sequence. Returns (T,) integer array."""
    T, K = log_emit.shape
    delta = np.empty((T, K))
    psi = np.empty((T, K), dtype=np.int32)
    delta[0] = log_pi + log_emit[0]
    psi[0] = 0
    for t in range(1, T):
        scores = delta[t - 1, :, None] + log_trans[t - 1]  # (K, K)
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = log_emit[t] + np.max(scores, axis=0)
    path = np.empty(T, dtype=np.int32)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path

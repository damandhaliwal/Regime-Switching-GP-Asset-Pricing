"""GP with ARD SE kernel: marginal log-likelihood and hyperparameter fitting.

A GP per regime. Hyperparameters per instance: log-lengthscales (D,),
log-signal-std, log-noise-std. We work in log-space so L-BFGS-B sees an
unconstrained problem; the positivity constraint is enforced by exp.

Marginal log-likelihood (for regime k):
    log p(y | X) = -1/2 y^T A^{-1} y  -  1/2 log|A|  -  N/2 log(2π)
    A = K(X, X; θ) + σ_n² I

Weighted variant (for M-step):
    L_w(θ) = Σ_t w_t * log p(r_t | X_t, θ),   w_t = γ_{t,k}
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from models.gp_jax import fit_gp_hyperparameters_jax

JITTER_BASE = 1e-6
JITTER_STEPS = (1.0, 10.0, 100.0, 1000.0)  # multiplied by JITTER_BASE


# --------------------------------------------------------------------------- #
# Kernel + marginal likelihood                                                #
# --------------------------------------------------------------------------- #

def se_ard_kernel(X: np.ndarray, Z: np.ndarray, lengthscales: np.ndarray,
                  signal_var: float) -> np.ndarray:
    """Squared-exponential ARD kernel: signal_var * exp(-0.5 * Σ_d (Δx_d/ℓ_d)²)."""
    Xs = X / lengthscales
    Zs = Z / lengthscales
    d2 = (Xs * Xs).sum(axis=1)[:, None] + (Zs * Zs).sum(axis=1)[None, :] - 2.0 * Xs @ Zs.T
    np.maximum(d2, 0.0, out=d2)
    return signal_var * np.exp(-0.5 * d2)


def _stable_chol(A: np.ndarray) -> tuple[np.ndarray, bool]:
    n = A.shape[0]
    I = np.eye(n)
    for mult in JITTER_STEPS:
        try:
            L = np.linalg.cholesky(A + JITTER_BASE * mult * I)
            return L, True
        except np.linalg.LinAlgError:
            continue
    return np.empty((n, n)), False


def marginal_log_likelihood(X: np.ndarray, y: np.ndarray,
                            lengthscales: np.ndarray,
                            signal_var: float, noise_var: float) -> float:
    N = y.shape[0]
    K = se_ard_kernel(X, X, lengthscales, signal_var)
    A = K + noise_var * np.eye(N)
    L, ok = _stable_chol(A)
    if not ok:
        return -np.inf
    alpha = cho_solve((L, True), y)
    log_det = 2.0 * np.log(np.diag(L)).sum()
    return -0.5 * y @ alpha - 0.5 * log_det - 0.5 * N * np.log(2.0 * np.pi)


# --------------------------------------------------------------------------- #
# Hyperparameter fitting                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class GPHyperparams:
    lengthscales: np.ndarray   # (D,)
    signal_var: float
    noise_var: float
    diagnostics: dict[str, object] | None = None

    def pack(self) -> np.ndarray:
        """Pack to unconstrained log-space vector: [log ℓ_1..D, log σ_f, log σ_n]."""
        return np.concatenate([
            np.log(self.lengthscales),
            [0.5 * np.log(self.signal_var), 0.5 * np.log(self.noise_var)],
        ])

    @classmethod
    def unpack(cls, v: np.ndarray, D: int) -> "GPHyperparams":
        return cls(
            lengthscales=np.exp(v[:D]),
            signal_var=float(np.exp(2.0 * v[D])),
            noise_var=float(np.exp(2.0 * v[D + 1])),
        )


def _neg_log_lik_weighted(v: np.ndarray, D: int,
                          data: Sequence[tuple[np.ndarray, np.ndarray]],
                          weights: np.ndarray,
                          prior: dict[str, np.ndarray | float] | None = None) -> float:
    """Sum over t of -w_t * log p(y_t | X_t, θ). v is the packed hyperparameter vector."""
    hp = GPHyperparams.unpack(v, D)
    total = 0.0
    for (X_t, y_t), w_t in zip(data, weights):
        if w_t <= 0 or y_t.size == 0:
            continue
        ll = marginal_log_likelihood(X_t, y_t, hp.lengthscales, hp.signal_var, hp.noise_var)
        if not np.isfinite(ll):
            return 1e10
        total += w_t * ll
    if prior is not None:
        log_ls = np.log(hp.lengthscales)
        log_sig = 0.5 * np.log(hp.signal_var)
        log_noise = 0.5 * np.log(hp.noise_var)
        total -= 0.5 * float(prior["lengthscale_weight"]) * float(
            np.sum(((log_ls - prior["lengthscale_center"]) / prior["lengthscale_scale"]) ** 2)
        )
        total -= 0.5 * float(prior["signal_weight"]) * float(
            ((log_sig - prior["signal_center"]) / prior["signal_scale"]) ** 2
        )
        total -= 0.5 * float(prior["noise_weight"]) * float(
            ((log_noise - prior["noise_center"]) / prior["noise_scale"]) ** 2
        )
    return -total


def _pooled_feature_scale(data: Sequence[tuple[np.ndarray, np.ndarray]], D: int) -> np.ndarray:
    x_all = np.vstack([X_t for X_t, _ in data if X_t.size])
    if x_all.size == 0:
        return np.ones(D)
    scale = x_all.std(axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    return scale.astype(np.float64)


def _pooled_response_scale(
    data: Sequence[tuple[np.ndarray, np.ndarray]],
    weights: np.ndarray,
) -> float:
    ys = [y_t for (_, y_t), w_t in zip(data, weights) if w_t > 0 and y_t.size]
    if not ys:
        return 1.0
    y_all = np.concatenate(ys)
    scale = float(np.std(y_all))
    return scale if np.isfinite(scale) and scale > 1e-4 else 1.0


def _empirical_lengthscale_seed(
    data: Sequence[tuple[np.ndarray, np.ndarray]],
    D: int,
) -> np.ndarray:
    x_all = np.vstack([X_t for X_t, _ in data if X_t.size])
    if x_all.shape[0] < 2:
        return np.ones(D)
    med = np.median(np.abs(x_all - np.median(x_all, axis=0, keepdims=True)), axis=0)
    med = np.where(np.isfinite(med) & (med > 1e-3), med, 1.0)
    return med.astype(np.float64)


def fit_gp_hyperparameters(
    data: Sequence[tuple[np.ndarray, np.ndarray]],
    weights: np.ndarray | None = None,
    D: int | None = None,
    n_restarts: int = 4,
    init: GPHyperparams | None = None,
    seed: int = 0,
) -> tuple[GPHyperparams, float]:
    """Fit GP hyperparameters by L-BFGS-B with random restarts.

    data: sequence of (X_t, y_t) pairs. weights: array of length len(data) or None (equal weights).
    Returns (best_params, best_neg_log_lik).
    """
    if D is None:
        D = data[0][0].shape[1]
    if weights is None:
        weights = np.ones(len(data))
    else:
        weights = np.asarray(weights, dtype=np.float64)

    feature_scale = _pooled_feature_scale(data, D)
    data_scaled = [(X_t / feature_scale, y_t.astype(np.float64)) for X_t, y_t in data]
    response_scale = _pooled_response_scale(data_scaled, weights)
    empirical_ls = _empirical_lengthscale_seed(data_scaled, D)

    rng = np.random.default_rng(seed)
    inits: list[np.ndarray] = []
    base_raw = init or GPHyperparams(lengthscales=np.ones(D), signal_var=1.0, noise_var=0.1)
    base = GPHyperparams(
        lengthscales=np.clip(base_raw.lengthscales / feature_scale, 1e-3, None),
        signal_var=base_raw.signal_var,
        noise_var=base_raw.noise_var,
    )
    signal_ref = max(0.85 * response_scale, 1e-3)
    noise_ref = max(0.10 * response_scale, 1e-4)
    # Prior weights are frozen (not scaled by number of active months) so the
    # regularizer does not strengthen as the rolling training window grows.
    # Centers are in *std* units (log σ_f, log σ_n), matching `pack()`.
    prior = {
        "lengthscale_center": np.log(np.clip(empirical_ls, 1e-3, 1e3)),
        "lengthscale_scale": np.full(D, 1.0),
        "lengthscale_weight": 1.0,
        "signal_center": float(np.log(signal_ref)),
        "signal_scale": 1.25,
        "signal_weight": 1.0,
        "noise_center": float(np.log(noise_ref)),
        "noise_scale": 0.75,
        "noise_weight": 5.0,
    }

    seed_params = [base]
    for hp0 in seed_params:
        inits.append(hp0.pack())
    for _ in range(n_restarts):
        center = inits[rng.integers(len(inits))]
        v = center + rng.normal(0.0, 0.35, size=D + 2)
        inits.append(v)

    # Sensible box on log-space: lengthscales in [1e-2, 1e2], stds in [1e-4, 1e2].
    bounds = [(np.log(1e-2), np.log(1e2))] * D + [(np.log(1e-4), np.log(1e2))] * 2

    best_v, best_f = fit_gp_hyperparameters_jax(
        data=data,
        weights=weights,
        D=D,
        init_log_params=inits[0],
        other_inits_log_params=inits[1:],
        prior=prior,
        bounds=bounds,
        feature_scale=feature_scale,
    )
    best_restart = -1  # JAX backend reports only the winning restart's loss

    if not np.isfinite(best_f):
        # Fall back to the base init unchanged.
        fallback = GPHyperparams(
            lengthscales=base.lengthscales * feature_scale,
            signal_var=base.signal_var,
            noise_var=base.noise_var,
            diagnostics={
                "feature_scale": feature_scale,
                "response_scale": response_scale,
                "signal_ref": signal_ref,
                "noise_ref": noise_ref,
                "best_restart": -1,
            },
        )
        return fallback, float(_neg_log_lik_weighted(base.pack(), D, data_scaled, weights, prior))
    best_hp = GPHyperparams.unpack(best_v, D)
    return GPHyperparams(
        lengthscales=best_hp.lengthscales * feature_scale,
        signal_var=best_hp.signal_var,
        noise_var=best_hp.noise_var,
        diagnostics={
            "feature_scale": feature_scale,
            "response_scale": response_scale,
            "signal_ref": signal_ref,
            "noise_ref": noise_ref,
            "best_restart": best_restart,
            "n_candidates": len(inits),
            "objective": best_f,
        },
    ), best_f


# --------------------------------------------------------------------------- #
# Posterior prediction                                                        #
# --------------------------------------------------------------------------- #

def predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
            hp: GPHyperparams) -> tuple[np.ndarray, np.ndarray]:
    """Return posterior (mean, variance) at X_test. Variance includes noise."""
    K = se_ard_kernel(X_train, X_train, hp.lengthscales, hp.signal_var)
    K += hp.noise_var * np.eye(X_train.shape[0])
    L, ok = _stable_chol(K)
    if not ok:
        N = X_test.shape[0]
        return np.zeros(N), np.full(N, hp.signal_var + hp.noise_var)
    alpha = cho_solve((L, True), y_train)
    Ks = se_ard_kernel(X_train, X_test, hp.lengthscales, hp.signal_var)
    mean = Ks.T @ alpha
    v = cho_solve((L, True), Ks)
    var_f = hp.signal_var - np.einsum("ij,ij->j", Ks, v)
    return mean, np.maximum(var_f, 0.0) + hp.noise_var

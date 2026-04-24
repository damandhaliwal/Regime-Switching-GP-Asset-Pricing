"""JAX backend for the GP M-step.

Public entry point: `fit_gp_hyperparameters_jax(...)`. The numpy shim in
`models/gp.py` wraps this and preserves the historical API.

Why JAX:
- JIT compiles the weighted marginal log-likelihood into one XLA kernel,
  removing the Python per-month loop that dominates scipy+numpy fits.
- `jax.value_and_grad` replaces scipy's finite-difference gradient
  (12 forward evals per L-BFGS step at D+2 = 12 params) with a single
  backward pass.
- Batched Cholesky over the (T, N, N) stack uses vectorized LAPACK.

Device: jax-metal 0.1.1 is pinned to jax 0.4.26 but the plugin does not
implement `mhlo.cholesky`, so we run on CPU JAX with float64 enabled.
Set `RSGP_JAX_BACKEND=cpu` (default) or `metal` (will fall back to cpu
on Cholesky lowering failure).

Ragged panels: each month may have a different number of stocks. We pad
every month to a common `N_max` with zeros, carry a boolean mask, and
patch the kernel so padded rows act as an identity block in the
Cholesky (contributes 0 to log-det, 0 to y^T K^{-1} y).
"""
from __future__ import annotations

import os
from typing import Sequence

import numpy as np

import jax

_backend = os.environ.get("RSGP_JAX_BACKEND", "cpu")
try:
    jax.config.update("jax_platform_name", _backend)
    _ = jax.devices()
except RuntimeError:
    jax.config.update("jax_platform_name", "cpu")

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit, value_and_grad, vmap
from jax.scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

JITTER_BASE = 1e-6
JITTER_STEPS = (1.0, 10.0, 100.0, 1000.0)


# --------------------------------------------------------------------------- #
# Kernel                                                                      #
# --------------------------------------------------------------------------- #

def _se_ard(X: jnp.ndarray, Z: jnp.ndarray, ls: jnp.ndarray, sf2: jnp.ndarray) -> jnp.ndarray:
    Xs = X / ls
    Zs = Z / ls
    d2 = (Xs * Xs).sum(axis=-1)[..., :, None] + (Zs * Zs).sum(axis=-1)[..., None, :] - 2.0 * Xs @ Zs.T
    d2 = jnp.maximum(d2, 0.0)
    return sf2 * jnp.exp(-0.5 * d2)


# --------------------------------------------------------------------------- #
# Single-month marginal log-likelihood with mask + jitter escalation          #
# --------------------------------------------------------------------------- #

def _chol_with_jitter(A: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Cholesky with escalating jitter. Returns (L, ok_flag)."""
    N = A.shape[0]
    I = jnp.eye(N, dtype=A.dtype)

    def try_chol(mult):
        M = A + JITTER_BASE * mult * I
        L = jnp.linalg.cholesky(M)
        ok = jnp.all(jnp.isfinite(L))
        return L, ok

    # Try escalating jitter.
    L, ok = try_chol(JITTER_STEPS[0])
    for mult in JITTER_STEPS[1:]:
        L_new, ok_new = try_chol(mult)
        L = jnp.where(ok, L, L_new)
        ok = ok | ok_new
    return L, ok


def _month_neg_ll(ls: jnp.ndarray, sf2: jnp.ndarray, sn2: jnp.ndarray,
                  X: jnp.ndarray, y: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Negative log-marginal for one padded month.

    X: (N, D), y: (N,), mask: (N,) with 1 on valid rows, 0 on pad.
    Padded rows are stitched into the kernel as an identity block (1 on
    diagonal, 0 elsewhere) so Cholesky sees a well-conditioned matrix
    and the padded block contributes 0 to both log-det and y^T K^{-1} y.
    """
    N = X.shape[0]
    I = jnp.eye(N, dtype=X.dtype)
    K_valid = _se_ard(X, X, ls, sf2) + sn2 * I
    M = mask[:, None] * mask[None, :]
    K = K_valid * M + (1.0 - M) * I
    y_masked = y * mask

    L, _ = _chol_with_jitter(K)
    alpha = jax.scipy.linalg.solve_triangular(L, y_masked, lower=True)
    quad = (alpha * alpha).sum()
    log_det = 2.0 * jnp.log(jnp.diag(L)).sum()
    n_valid = mask.sum()
    return 0.5 * quad + 0.5 * log_det + 0.5 * n_valid * jnp.log(2.0 * jnp.pi)


def _prior_penalty(v: jnp.ndarray, D: int, prior: dict) -> jnp.ndarray:
    log_ls = v[:D]
    log_sig = v[D]
    log_noise = v[D + 1]

    pen = 0.5 * prior["lengthscale_weight"] * jnp.sum(
        ((log_ls - prior["lengthscale_center"]) / prior["lengthscale_scale"]) ** 2
    )
    pen = pen + 0.5 * prior["signal_weight"] * (
        ((log_sig - prior["signal_center"]) / prior["signal_scale"]) ** 2
    )
    pen = pen + 0.5 * prior["noise_weight"] * (
        ((log_noise - prior["noise_center"]) / prior["noise_scale"]) ** 2
    )
    return pen


def _neg_ll_total(v: jnp.ndarray,
                  X_stack: jnp.ndarray, y_stack: jnp.ndarray,
                  mask_stack: jnp.ndarray, weights: jnp.ndarray,
                  ls_center: jnp.ndarray, ls_scale: jnp.ndarray, ls_weight: jnp.ndarray,
                  sig_center: jnp.ndarray, sig_scale: jnp.ndarray, sig_weight: jnp.ndarray,
                  noise_center: jnp.ndarray, noise_scale: jnp.ndarray, noise_weight: jnp.ndarray,
                  ) -> jnp.ndarray:
    D = ls_center.shape[0]
    ls = jnp.exp(v[:D])
    sf2 = jnp.exp(2.0 * v[D])
    sn2 = jnp.exp(2.0 * v[D + 1])

    per_month = vmap(lambda X_t, y_t, m_t: _month_neg_ll(ls, sf2, sn2, X_t, y_t, m_t))(
        X_stack, y_stack, mask_stack
    )
    weighted = (weights * per_month).sum()

    log_ls = v[:D]
    log_sig = v[D]
    log_noise = v[D + 1]
    pen = 0.5 * ls_weight * jnp.sum(((log_ls - ls_center) / ls_scale) ** 2)
    pen = pen + 0.5 * sig_weight * ((log_sig - sig_center) / sig_scale) ** 2
    pen = pen + 0.5 * noise_weight * ((log_noise - noise_center) / noise_scale) ** 2
    return weighted + pen


_value_and_grad_fn = jit(value_and_grad(_neg_ll_total))


# --------------------------------------------------------------------------- #
# Public fit API                                                              #
# --------------------------------------------------------------------------- #

def _stack_and_pad(data: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad every month to the max N across months; build mask."""
    sizes = [y.size for _, y in data]
    N_max = int(max(sizes))
    T = len(data)
    D = data[0][0].shape[1]
    X_stack = np.zeros((T, N_max, D), dtype=np.float64)
    y_stack = np.zeros((T, N_max), dtype=np.float64)
    mask = np.zeros((T, N_max), dtype=np.float64)
    for t, (X_t, y_t) in enumerate(data):
        n = y_t.size
        X_stack[t, :n] = X_t
        y_stack[t, :n] = y_t
        mask[t, :n] = 1.0
    return X_stack, y_stack, mask


def fit_gp_hyperparameters_jax(
    data: Sequence[tuple[np.ndarray, np.ndarray]],
    weights: np.ndarray,
    D: int,
    init_log_params: np.ndarray,
    other_inits_log_params: Sequence[np.ndarray],
    prior: dict,
    bounds: list[tuple[float, float]],
    feature_scale: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Fit the packed log-hyperparameter vector on JAX.

    Returns (best_v_log, best_neg_log_lik). `v` is in *scaled* feature space;
    caller multiplies lengthscales by `feature_scale` to recover data-space
    values.
    """
    # Scale features once (numpy) and stack for JAX.
    data_scaled = [(X_t / feature_scale, y_t.astype(np.float64)) for X_t, y_t in data]
    X_stack_np, y_stack_np, mask_np = _stack_and_pad(data_scaled)

    X_stack = jnp.asarray(X_stack_np)
    y_stack = jnp.asarray(y_stack_np)
    mask_stack = jnp.asarray(mask_np)
    w = jnp.asarray(np.asarray(weights, dtype=np.float64))

    ls_center = jnp.asarray(np.asarray(prior["lengthscale_center"], dtype=np.float64))
    ls_scale = jnp.asarray(np.asarray(prior["lengthscale_scale"], dtype=np.float64))
    ls_weight = jnp.asarray(float(prior["lengthscale_weight"]))
    sig_center = jnp.asarray(float(prior["signal_center"]))
    sig_scale = jnp.asarray(float(prior["signal_scale"]))
    sig_weight = jnp.asarray(float(prior["signal_weight"]))
    noise_center = jnp.asarray(float(prior["noise_center"]))
    noise_scale = jnp.asarray(float(prior["noise_scale"]))
    noise_weight = jnp.asarray(float(prior["noise_weight"]))

    def fg(v_np: np.ndarray) -> tuple[float, np.ndarray]:
        loss, grad = _value_and_grad_fn(
            jnp.asarray(v_np),
            X_stack, y_stack, mask_stack, w,
            ls_center, ls_scale, ls_weight,
            sig_center, sig_scale, sig_weight,
            noise_center, noise_scale, noise_weight,
        )
        return float(loss), np.asarray(grad, dtype=np.float64)

    best_v = None
    best_f = np.inf
    for v0 in [init_log_params, *other_inits_log_params]:
        try:
            res = minimize(
                fg, v0,
                method="L-BFGS-B", jac=True, bounds=bounds,
                options={"maxiter": 100, "gtol": 1e-5, "ftol": 1e-8},
            )
            if np.isfinite(res.fun) and res.fun < best_f:
                best_f = float(res.fun)
                best_v = res.x
        except Exception:
            continue

    if best_v is None:
        return np.asarray(init_log_params, dtype=np.float64), float(np.inf)
    return np.asarray(best_v, dtype=np.float64), float(best_f)
